from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import modules.compat as compat
from basics.base_module import CategorizedModule
from modules.aux_decoder import AuxDecoderAdaptor
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
    NormalInitEmbedding as Embedding
)
from modules.core import (
    GaussianDiffusion, PitchDiffusion, MultiVarianceDiffusion,
    RectifiedFlow, PitchRectifiedFlow, MultiVarianceRectifiedFlow
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.param_adaptor import ParameterAdaptorModule
from modules.fastspeech.tts_modules import RhythmRegulator, LengthRegulator, mel2ph_to_dur
from modules.fastspeech.variance_encoder import FastSpeech2Variance, MelodyEncoder
from utils.hparams import hparams


class ShallowDiffusionOutput:
    def __init__(self, *, aux_out=None, diff_out=None, alf_out=None):
        self.aux_out = aux_out
        self.diff_out = diff_out
        # alf_out: (attn_softs, attn_hards, attn_logprobs, token_lengths, mel_lengths)
        # or None when ALF is not used
        self.alf_out = alf_out


class DiffSingerAcoustic(CategorizedModule, ParameterAdaptorModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims):
        CategorizedModule.__init__(self)
        ParameterAdaptorModule.__init__(self)
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )

        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        self.shallow_args = hparams.get('shallow_diffusion_args', {})
        if self.use_shallow_diffusion:
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']
            self.aux_decoder_grad = self.shallow_args['aux_decoder_grad']
            self.aux_decoder = AuxDecoderAdaptor(
                in_dims=hparams['hidden_size'], out_dims=out_dims, num_feats=1,
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
                aux_decoder_arch=self.shallow_args['aux_decoder_arch'],
                aux_decoder_args=self.shallow_args['aux_decoder_args']
            )
        self.diffusion_type = hparams.get('diffusion_type', 'ddpm')
        self.backbone_type = compat.get_backbone_type(hparams)
        self.backbone_args = compat.get_backbone_args(hparams, self.backbone_type)
        if self.diffusion_type == 'ddpm':
            self.diffusion = GaussianDiffusion(
                out_dims=out_dims,
                num_feats=1,
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        elif self.diffusion_type == 'reflow':
            self.diffusion = RectifiedFlow(
                out_dims=out_dims,
                num_feats=1,
                t_start=hparams['T_start'],
                time_scale_factor=hparams['time_scale_factor'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        else:
            raise NotImplementedError(self.diffusion_type)

        # Alignment Learning Framework (ALF) for training without duration annotations
        self.use_alf = hparams.get('use_alf', False)
        if self.use_alf:
            from modules.alignment import (
                Alignment_Learning_Framework,
                BetaBinomialInterpolator,
            )
            self.alf = Alignment_Learning_Framework(
                feature_size=hparams['audio_num_mel_bins'],
                encoding_size=hparams['hidden_size']
            )
            self._alf_prior_interpolator = BetaBinomialInterpolator()
            self._alf_lr = LengthRegulator()

    def _compute_attn_priors(self, mel_lengths, token_lengths, max_mel_len, max_token_len, device):
        """Compute beta-binomial attention priors for the batch."""
        B = mel_lengths.shape[0]
        priors = torch.zeros(B, max_mel_len, max_token_len, device=device)
        for b in range(B):
            m_len = mel_lengths[b].item()
            t_len = token_lengths[b].item()
            if m_len > 0 and t_len > 0:
                prior = self._alf_prior_interpolator(m_len, t_len)  # [m_len, t_len]
                priors[b, :m_len, :t_len] = torch.from_numpy(prior).to(device)
        return priors  # [B, T_mel, T_text]

    def forward(
            self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
            spk_embed_id=None, languages=None, gt_mel=None, infer=True,
            needs_alignment=None, mel_lengths=None, **kwargs
    ) -> ShallowDiffusionOutput:
        """
        Args:
            needs_alignment: [B] bool tensor; True for items that need ALF-based alignment
            mel_lengths: [B] int tensor; actual mel frame counts (required when needs_alignment is set)
        """
        alf_out = None

        # Use ALF when there are items without duration annotations (training or validation)
        if (
            self.use_alf
            and needs_alignment is not None
            and needs_alignment.any()
            and gt_mel is not None
        ):
            # Compute per-item token lengths (non-padding tokens)
            token_lengths = (txt_tokens != 0).sum(dim=1)  # [B]

            if mel_lengths is None:
                # Fall back to counting non-zero mel2ph entries (works for GT items;
                # for ALF items mel2ph is placeholder zeros, but mel_lengths must be passed)
                mel_lengths = (mel2ph > 0).sum(dim=1)
                mel_lengths = torch.clamp(mel_lengths, min=1)

            # Compute dur for the encoder:
            #   GT items  → use GT mel2ph to compute dur (preserves dur conditioning)
            #   ALF items → use zero dur (alignment will be learned)
            dur_gt = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
            dur = dur_gt * (~needs_alignment[:, None]).float()

            # Single encoder pass
            encoder_out = self.fs2.forward_encoder(txt_tokens, languages=languages, dur=dur)

            # Compute beta-binomial attention priors
            max_mel_len = gt_mel.shape[1]
            max_token_len = txt_tokens.shape[1]
            attn_priors = self._compute_attn_priors(
                mel_lengths, token_lengths, max_mel_len, max_token_len, txt_tokens.device
            )

            # Run ALF: obtain soft/hard alignments
            durations, attn_softs, attn_hards, attn_logprobs = self.alf(
                token_embeddings=encoder_out.transpose(1, 2),  # [B, H, T_text]
                encoding_lengths=token_lengths,
                features=gt_mel.transpose(1, 2),               # [B, n_mel, T_mel]
                feature_lengths=mel_lengths,
                attention_priors=attn_priors
            )

            # Convert hard-alignment durations to mel2ph
            mel2ph_alf = self._alf_lr(durations.long())  # [B, T_mel_alf]

            # Pad/crop to match gt_mel length
            if mel2ph_alf.shape[1] < max_mel_len:
                mel2ph_alf = F.pad(mel2ph_alf, [0, max_mel_len - mel2ph_alf.shape[1]])
            else:
                mel2ph_alf = mel2ph_alf[:, :max_mel_len]

            # Replace mel2ph only for ALF items (keep GT mel2ph for annotated items)
            mel2ph = torch.where(needs_alignment[:, None], mel2ph_alf, mel2ph)

            # Gather condition from shared encoder_out
            condition = self.fs2.forward_gather(
                encoder_out, mel2ph, f0,
                key_shift=key_shift, speed=speed,
                spk_embed_id=spk_embed_id, **kwargs
            )

            alf_out = (attn_softs, attn_hards, attn_logprobs, token_lengths, mel_lengths)
        else:
            condition = self.fs2(
                txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
                spk_embed_id=spk_embed_id, languages=languages,
                **kwargs
            )

        if infer:
            if self.use_shallow_diffusion:
                aux_mel_pred = self.aux_decoder(condition, infer=True)
                aux_mel_pred *= ((mel2ph > 0).float()[:, :, None])
                if gt_mel is not None and self.shallow_args['val_gt_start']:
                    src_mel = gt_mel
                else:
                    src_mel = aux_mel_pred
            else:
                aux_mel_pred = src_mel = None
            mel_pred = self.diffusion(condition, src_spec=src_mel, infer=True)
            mel_pred *= ((mel2ph > 0).float()[:, :, None])
            return ShallowDiffusionOutput(aux_out=aux_mel_pred, diff_out=mel_pred)
        else:
            if self.use_shallow_diffusion:
                if self.train_aux_decoder:
                    aux_cond = condition * self.aux_decoder_grad + condition.detach() * (1 - self.aux_decoder_grad)
                    aux_out = self.aux_decoder(aux_cond, infer=False)
                else:
                    aux_out = None
                if self.train_diffusion:
                    diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
                else:
                    diff_out = None
                return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out, alf_out=alf_out)

            else:
                aux_out = None
                diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
                return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out, alf_out=alf_out)


class DiffSingerVariance(CategorizedModule, ParameterAdaptorModule):
    @property
    def category(self):
        return 'variance'

    def __init__(self, vocab_size):
        CategorizedModule.__init__(self)
        ParameterAdaptorModule.__init__(self)
        self.predict_dur = hparams['predict_dur']
        self.predict_pitch = hparams['predict_pitch']

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.rr = RhythmRegulator()
        self.lr = LengthRegulator()
        self.diffusion_type = hparams.get('diffusion_type', 'ddpm')
        if self.predict_pitch:
            self.use_melody_encoder = hparams.get('use_melody_encoder', False)
            if self.use_melody_encoder:
                self.melody_encoder = MelodyEncoder(enc_hparams=hparams['melody_encoder_args'])
                self.delta_pitch_embed = Linear(1, hparams['hidden_size'])
            else:
                self.base_pitch_embed = Linear(1, hparams['hidden_size'])

            self.pitch_retake_embed = Embedding(2, hparams['hidden_size'])
            pitch_hparams = hparams['pitch_prediction_args']
            self.pitch_backbone_type = compat.get_backbone_type(hparams, nested_config=pitch_hparams)
            self.pitch_backbone_args = compat.get_backbone_args(pitch_hparams, backbone_type=self.pitch_backbone_type)
            if self.diffusion_type == 'ddpm':
                self.pitch_predictor = PitchDiffusion(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    timesteps=hparams['timesteps'],
                    k_step=hparams['K_step'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            elif self.diffusion_type == 'reflow':
                self.pitch_predictor = PitchRectifiedFlow(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    time_scale_factor=hparams['time_scale_factor'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            else:
                raise ValueError(f"Invalid diffusion type: {self.diffusion_type}")

        if self.predict_variances:
            self.pitch_embed = Linear(1, hparams['hidden_size'])
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_prediction_list
            })

            if self.diffusion_type == 'ddpm':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceDiffusion)
            elif self.diffusion_type == 'reflow':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceRectifiedFlow)
            else:
                raise NotImplementedError(self.diffusion_type)

    def forward(
            self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, mel2ph=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None, mel2note=None,
            base_pitch=None, pitch=None, pitch_expr=None, pitch_retake=None,
            variance_retake: Dict[str, Tensor] = None,
            spk_id=None, languages=None,
            infer=True, **kwargs
    ):
        if self.use_spk_id:
            ph_spk_mix_embed = kwargs.get('ph_spk_mix_embed')
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if ph_spk_mix_embed is not None and spk_mix_embed is not None:
                ph_spk_embed = ph_spk_mix_embed
                spk_embed = spk_mix_embed
            else:
                ph_spk_embed = spk_embed = self.spk_embed(spk_id)[:, None, :]  # [B,] => [B, T=1, H]
        else:
            ph_spk_embed = spk_embed = None

        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur,
            spk_embed=ph_spk_embed, languages=languages,
            infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, ({} if infer else None)

        if mel2ph is None and word_dur is not None:  # inference from file
            dur_pred_align = self.rr(dur_pred_out, ph2word, word_dur)
            mel2ph = self.lr(dur_pred_align)
            mel2ph = F.pad(mel2ph, [0, base_pitch.shape[1] - mel2ph.shape[1]])

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.use_spk_id:
            condition += spk_embed

        if self.predict_pitch:
            if self.use_melody_encoder:
                melody_encoder_out = self.melody_encoder(
                    note_midi, note_rest, note_dur,
                    glide=note_glide
                )
                melody_encoder_out = F.pad(melody_encoder_out, [0, 0, 1, 0])
                mel2note_ = mel2note[..., None].repeat([1, 1, hparams['hidden_size']])
                melody_condition = torch.gather(melody_encoder_out, 1, mel2note_)
                pitch_cond = condition + melody_condition
            else:
                pitch_cond = condition.clone()  # preserve the original tensor to avoid further inplace operations

            retake_unset = pitch_retake is None
            if retake_unset:
                pitch_retake = torch.ones_like(mel2ph, dtype=torch.bool)

            if pitch_expr is None:
                pitch_retake_embed = self.pitch_retake_embed(pitch_retake.long())
            else:
                retake_true_embed = self.pitch_retake_embed(
                    torch.ones(1, 1, dtype=torch.long, device=txt_tokens.device)
                )  # [B=1, T=1] => [B=1, T=1, H]
                retake_false_embed = self.pitch_retake_embed(
                    torch.zeros(1, 1, dtype=torch.long, device=txt_tokens.device)
                )  # [B=1, T=1] => [B=1, T=1, H]
                pitch_expr = (pitch_expr * pitch_retake)[:, :, None]  # [B, T, 1]
                pitch_retake_embed = pitch_expr * retake_true_embed + (1. - pitch_expr) * retake_false_embed

            pitch_cond += pitch_retake_embed
            if self.use_melody_encoder:
                if retake_unset:  # generate from scratch
                    delta_pitch_in = torch.zeros_like(base_pitch)
                else:
                    delta_pitch_in = (pitch - base_pitch) * ~pitch_retake
                pitch_cond += self.delta_pitch_embed(delta_pitch_in[:, :, None])
            else:
                if not retake_unset:  # retake
                    base_pitch = base_pitch * pitch_retake + pitch * ~pitch_retake
                pitch_cond += self.base_pitch_embed(base_pitch[:, :, None])

            if infer:
                pitch_pred_out = self.pitch_predictor(pitch_cond, infer=True)
            else:
                pitch_pred_out = self.pitch_predictor(pitch_cond, pitch - base_pitch, infer=False)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None)

        if pitch is None:
            pitch = base_pitch + pitch_pred_out
        var_cond = condition + self.pitch_embed(pitch[:, :, None])

        variance_inputs = self.collect_variance_inputs(**kwargs)
        if variance_retake is not None:
            variance_embeds = [
                self.variance_embeds[v_name](v_input[:, :, None]) * ~variance_retake[v_name][:, :, None]
                for v_name, v_input in zip(self.variance_prediction_list, variance_inputs)
            ]
            var_cond += torch.stack(variance_embeds, dim=-1).sum(-1)

        variance_outputs = self.variance_predictor(var_cond, variance_inputs, infer=infer)

        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out
