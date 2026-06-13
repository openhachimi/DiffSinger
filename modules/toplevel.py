import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config.schema import ModelConfig
from .alignment import AlignmentLearningFramework, BetaBinomialInterpolator
from .bbc_mask import fast_fast_bbc_mask
from .commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from .commons.tts_modules import LocalUpsample
from .decoder import DiffusionDecoder, ShallowDiffusionOutput
from .embedding import ParameterEmbeddings
from .encoder import LinguisticEncoder, MelodyEncoder
from .normalizer import FeatureNormalizer

__all__ = [
    "DiffSingerAcoustic",
    "DiffSingerVariance",
]


class DiffSingerAcoustic(nn.Module):
    def __init__(self, config: ModelConfig, alf_pad_token_id: int | None = None):
        super().__init__()
        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()  # tokens to frames
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.speaker_embedding = Embedding(config.num_spk, config.condition_dim)
        self.parameter_embeddings = ParameterEmbeddings(config=config.embeddings)
        self.spec_decoder = DiffusionDecoder(
            sample_dim=config.sample_dim,
            condition_dim=config.condition_dim,
            normalizer=FeatureNormalizer(
                num_channels=config.sample_dim, num_features=1, num_repeats=None,
                squeeze_channel_dim=False, squeeze_feature_dim=True,
                norm_mins=[config.normalization.spec_min],
                norm_maxs=[config.normalization.spec_max],
            ),
            config=config.spec_decoder
        )
        self.use_alf = config.use_alf
        if self.use_alf:
            self.alf = AlignmentLearningFramework(
                feature_size=config.sample_dim,
                encoding_size=config.condition_dim
            )
            self.alf_prior_interpolator = BetaBinomialInterpolator()
            self.alf_use_interleaved_pad = config.alf_use_interleaved_pad
            self.alf_pad_token_id = alf_pad_token_id
            if self.alf_use_interleaved_pad and (self.alf_pad_token_id is None or self.alf_pad_token_id <= 0):
                raise ValueError(
                    "Invalid ALF pad token id. Set model.alf_pad_phoneme to a valid global phoneme."
                )
        else:
            self.alf_use_interleaved_pad = False
        self.use_bbc_encoder = config.use_bbc_encoder
        if self.use_bbc_encoder:
            self.bbc_mask_emb = nn.Parameter(torch.randn(1, 1, config.condition_dim))
            self.bbc_mask_len = config.bbc_mask_len
            self.bbc_min_segment_length = config.bbc_min_segment_length
            self.bbc_mask_prob = config.bbc_mask_prob

    @staticmethod
    def _expand_tokens_for_alf(tokens, needs_alignment, pad_token_id, languages=None):
        batch_tokens = []
        batch_languages = [] if languages is not None else None
        batch_maps = []
        batch_lengths = []
        batch_size = tokens.shape[0]
        device = tokens.device
        for b in range(batch_size):
            token_row = tokens[b]
            valid_mask = token_row > 0
            valid_tokens = token_row[valid_mask]
            token_count = valid_tokens.shape[0]
            if needs_alignment[b].item() and token_count > 0:
                expanded_len = token_count * 2 + 1
                expanded_tokens = token_row.new_full((expanded_len,), pad_token_id)
                expanded_tokens[1::2] = valid_tokens
                expanded_map = token_row.new_zeros((expanded_len,))
                expanded_map[1::2] = torch.arange(1, token_count + 1, device=device, dtype=token_row.dtype)
                if languages is not None:
                    language_row = languages[b]
                    valid_languages = language_row[valid_mask]
                    expanded_languages = language_row.new_zeros((expanded_len,))
                    expanded_languages[1::2] = valid_languages
            else:
                expanded_len = token_count
                expanded_tokens = valid_tokens
                expanded_map = token_row.new_zeros((expanded_len,))
                if token_count > 0:
                    expanded_map[:] = torch.arange(1, token_count + 1, device=device, dtype=token_row.dtype)
                if languages is not None:
                    language_row = languages[b]
                    expanded_languages = language_row[valid_mask]
            batch_tokens.append(expanded_tokens)
            batch_maps.append(expanded_map)
            if languages is not None:
                batch_languages.append(expanded_languages)
            batch_lengths.append(expanded_len)
        max_len = max(batch_lengths)
        expanded_tokens = tokens.new_zeros((batch_size, max_len))
        expanded_maps = tokens.new_zeros((batch_size, max_len))
        expanded_lengths = tokens.new_tensor(batch_lengths, dtype=torch.long)
        if languages is not None:
            expanded_languages = languages.new_zeros((batch_size, max_len))
        else:
            expanded_languages = None
        for b in range(batch_size):
            row_len = batch_lengths[b]
            if row_len > 0:
                expanded_tokens[b, :row_len] = batch_tokens[b]
                expanded_maps[b, :row_len] = batch_maps[b]
                if expanded_languages is not None:
                    expanded_languages[b, :row_len] = batch_languages[b]
        return expanded_tokens, expanded_languages, expanded_maps, expanded_lengths

    @staticmethod
    def _recover_no_pad_encoder_out(encoder_out, token_maps, target_len):
        batch_size, _, hidden_size = encoder_out.shape
        encoder_out_no_pad = encoder_out.new_zeros((batch_size, target_len, hidden_size))
        for b in range(batch_size):
            real_mask = token_maps[b] > 0
            token_count = int(real_mask.sum().item())
            if token_count > 0:
                encoder_out_no_pad[b, :token_count] = encoder_out[b, real_mask][:token_count]
        return encoder_out_no_pad

    def _recover_no_pad_durations(self, durations, token_maps, target_len):
        batch_size = durations.shape[0]
        durations_no_pad = durations.new_zeros((batch_size, target_len))
        for b in range(batch_size):
            real_mask = token_maps[b] > 0
            token_count = int(real_mask.sum().item())
            if token_count > 0:
                durations_no_pad[b, :token_count] = durations[b, real_mask][:token_count]
        return durations_no_pad

    def _compute_attn_priors(self, mel_lengths, token_lengths, max_mel_len, max_token_len, device):
        batch_size = mel_lengths.shape[0]
        priors = torch.zeros(batch_size, max_mel_len, max_token_len, device=device)
        for b in range(batch_size):
            mel_len = mel_lengths[b].item()
            token_len = token_lengths[b].item()
            if mel_len > 0 and token_len > 0:
                prior = self.alf_prior_interpolator(mel_len, token_len)
                priors[b, :mel_len, :token_len] = torch.from_numpy(prior).to(device)
        return priors

    def _upsample(self, encoder_out, durations):
        """Upsample encoder output from phoneme to frame level.

        When use_bbc_encoder is True, inserts a learnable blur-boundary token
        at encoder position 1 and replaces the tail frames of each phoneme
        segment with that token before gathering, so the model must infer the
        exact phoneme boundary from context.  The ``durations`` passed here
        should already reflect ALF-predicted values when ALF is active,
        satisfying the requirement to use the ALF output as the bbc_encoder
        ph_dur input.
        """
        if self.use_bbc_encoder:
            # Compute frame-level phoneme indices from durations
            mel2ph = self.local_upsample.lr(durations)  # [B, T_mel]
            # Apply BBC mask: shift non-padding indices by +1 and replace
            # tail frames of long-enough segments with index 1 (bbc_mask_emb)
            mel2ph = fast_fast_bbc_mask(
                mel2ph,
                mask_length=self.bbc_mask_len,
                min_segment_length=self.bbc_min_segment_length,
                mask_prob=self.bbc_mask_prob,
            )
            # Build augmented encoder output:
            # position 0: zero padding  (for mel2ph == 0)
            # position 1: bbc_mask_emb  (for BBC-masked boundary frames)
            # position 2+: phoneme encodings
            bbc_emb = self.bbc_mask_emb.expand(encoder_out.shape[0], 1, encoder_out.shape[-1])
            encoder_out_bbc = F.pad(
                torch.cat([bbc_emb, encoder_out], dim=1),
                [0, 0, 1, 0],
            )  # [B, T_ph+2, H]
            # Gather frame-level representations
            H = encoder_out_bbc.shape[-1]
            mel2ph_idx = mel2ph.unsqueeze(-1).expand(-1, -1, H)
            cond = torch.gather(encoder_out_bbc, 1, mel2ph_idx)
            mask = mel2ph > 0
            return cond, mask
        return self.local_upsample(encoder_out, ups=durations)

    def forward(
            self, tokens, durations, languages, f0, spk_ids=None,
            spk_embed=None, spec_gt=None, infer=True,
            needs_alignment=None, mel_lengths=None, **kwargs
    ) -> ShallowDiffusionOutput:
        alf_out = None
        if (
                self.use_alf and
                needs_alignment is not None and
                needs_alignment.any() and
                spec_gt is not None
        ):
            if mel_lengths is None:
                mel_lengths = (spec_gt.abs().sum(dim=-1) > 0).sum(dim=1)
            if self.alf_use_interleaved_pad:
                tokens_alf, languages_alf, token_maps, token_lengths = self._expand_tokens_for_alf(
                    tokens, needs_alignment, self.alf_pad_token_id, languages=languages
                )
                durations_alf = durations.new_zeros(tokens_alf.shape)
                non_alf_mask = ~needs_alignment
                if non_alf_mask.any():
                    durations_alf[non_alf_mask, :durations.shape[1]] = durations[non_alf_mask]
            else:
                tokens_alf = tokens
                languages_alf = languages
                token_maps = None
                token_lengths = (tokens_alf != 0).sum(dim=1)
                durations_alf = durations * (~needs_alignment[:, None]).long()
            encoder_out = self.linguistic_encoder(tokens=tokens_alf, durations=durations_alf, languages=languages_alf)
            max_mel_len = spec_gt.shape[1]
            max_token_len = tokens_alf.shape[1]
            attn_priors = self._compute_attn_priors(
                mel_lengths, token_lengths, max_mel_len, max_token_len, spec_gt.device
            )
            alf_durations, attn_softs, attn_hards, attn_logprobs = self.alf(
                token_embeddings=encoder_out.transpose(1, 2),
                encoding_lengths=token_lengths,
                features=spec_gt.transpose(1, 2),
                feature_lengths=mel_lengths,
                attention_priors=attn_priors
            )
            if self.alf_use_interleaved_pad:
                alf_durations = self._recover_no_pad_durations(alf_durations.long(), token_maps, durations.shape[1])
                encoder_out = self._recover_no_pad_encoder_out(encoder_out, token_maps, durations.shape[1])
            durations = torch.where(needs_alignment[:, None], alf_durations.long(), durations.long())
            alf_out = (attn_softs, attn_hards, attn_logprobs, token_lengths, mel_lengths)
        else:
            encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self._upsample(encoder_out, durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.speaker_embedding(spk_ids)[:, None, :]
            cond = cond + spk_embed
        cond = self.parameter_embeddings(cond, f0=f0, **kwargs)
        decoder_out = self.spec_decoder(condition=cond, sample_gt=spec_gt, infer=infer)
        decoder_out.alf_out = alf_out
        return decoder_out, mask


class DiffSingerVariance(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.predict_pitch = config.prediction.predict_pitch
        var_norm_mins = []
        var_norm_maxs = []
        var_clip_mins = []
        var_clip_maxs = []
        variance_list = config.prediction.predicted_variance_names
        if config.prediction.predict_energy:
            var_norm_mins.append(config.normalization.energy_db_min)
            var_norm_maxs.append(config.normalization.energy_db_max)
            var_clip_mins.append(config.normalization.energy_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_breathiness:
            var_norm_mins.append(config.normalization.breathiness_db_min)
            var_norm_maxs.append(config.normalization.breathiness_db_max)
            var_clip_mins.append(config.normalization.breathiness_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_voicing:
            var_norm_mins.append(config.normalization.voicing_db_min)
            var_norm_maxs.append(config.normalization.voicing_db_max)
            var_clip_mins.append(config.normalization.voicing_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_tension:
            var_norm_mins.append(config.normalization.tension_logit_min)
            var_norm_maxs.append(config.normalization.tension_logit_max)
            var_clip_mins.append(config.normalization.tension_logit_min)
            var_clip_maxs.append(config.normalization.tension_logit_max)
        self.predict_variances = len(variance_list) > 0
        self.variance_list = variance_list
        if not self.predict_pitch and not self.predict_variances:
            raise ValueError("Nothing to predict.")

        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.spk_embed = Embedding(config.num_spk, config.condition_dim)
        if self.predict_pitch:
            self.melody_encoder = MelodyEncoder(config=config.melody_encoder)
            self.pitch_predictor = DiffusionDecoder(
                sample_dim=config.normalization.pitch_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=1, num_repeats=config.normalization.pitch_repeat_bins,
                    squeeze_channel_dim=True, squeeze_feature_dim=True,
                    norm_mins=[config.normalization.pitd_norm_min],
                    norm_maxs=[config.normalization.pitd_norm_max],
                    clip_mins=[config.normalization.pitd_clip_min],
                    clip_maxs=[config.normalization.pitd_clip_max],
                ),
                config=config.pitch_predictor
            )
        if self.predict_variances:
            total_repeat_bins = config.normalization.variance_total_repeat_bins
            if total_repeat_bins % len(self.variance_list) != 0:
                raise ValueError(
                    f"variance_total_repeat_bins must be divisible by "
                    f"number of variances ({len(self.variance_list)})."
                )
            self.pitch_embedding = Linear(1, config.condition_dim)
            self.variance_predictor = DiffusionDecoder(
                sample_dim=total_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=len(self.variance_list),
                    num_repeats=total_repeat_bins // len(self.variance_list),
                    squeeze_channel_dim=True, squeeze_feature_dim=False,
                    norm_mins=var_norm_mins, norm_maxs=var_norm_maxs,
                    clip_mins=var_clip_mins, clip_maxs=var_clip_maxs,
                ),
                config=config.variance_predictor
            )

    def forward(
            self, tokens, durations, languages, spk_ids=None, spk_embed=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None,
            base_pitch=None, pitch=None,
            infer=True, **kwargs
    ):
        linguistic_encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self.local_upsample(linguistic_encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.spk_embed(spk_ids)[:, None, :]
            cond = cond + spk_embed
        if self.predict_pitch:
            melody_encoder_out = self.melody_encoder(
                note_midi=note_midi, note_rest=note_rest, note_dur=note_dur, glide=note_glide
            )
            # TODO: add pitch retaking and expressiveness
            pitch_cond = cond + self.local_upsample(melody_encoder_out, ups=note_dur)[0]
            pitch_predictor_out = self.pitch_predictor(condition=pitch_cond, sample_gt=pitch - base_pitch, infer=infer)
            pitch_predictor_out = pitch_predictor_out.diff_out  # no shallow diffusion yet
            if infer:
                pitch_predictor_out = pitch_predictor_out + base_pitch
        else:
            pitch_predictor_out = None
        if self.predict_variances:
            variance_cond = cond + self.pitch_embedding(pitch[:, :, None])
            variance_predictor_out = self.variance_predictor(
                condition=variance_cond,
                sample_gt=[kwargs.get(v_name) for v_name in self.variance_list],
                infer=infer
            )
            variance_predictor_out = variance_predictor_out.diff_out  # no shallow diffusion yet
        else:
            variance_predictor_out = None
        return pitch_predictor_out, variance_predictor_out, mask
