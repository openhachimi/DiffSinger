import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.aux_decoder import build_aux_loss
from modules.losses import DiffusionLoss, RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from utils.hparams import hparams
from utils.plot import spec_to_figure, alf_to_figure

matplotlib.use('Agg')


class AcousticDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(AcousticDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)
        self.required_variances = {}  # key: variance name, value: padding value
        if hparams['use_energy_embed']:
            self.required_variances['energy'] = 0.0
        if hparams['use_breathiness_embed']:
            self.required_variances['breathiness'] = 0.0
        if hparams['use_voicing_embed']:
            self.required_variances['voicing'] = 0.0
        if hparams['use_tension_embed']:
            self.required_variances['tension'] = 0.0

        self.need_key_shift = hparams['use_key_shift_embed']
        self.need_speed = hparams['use_speed_embed']
        self.need_spk_id = hparams['use_spk_id']
        self.need_lang_id = hparams['use_lang_id']
        self.use_alf = hparams.get('use_alf', False)

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        mel = utils.collate_nd([s['mel'] for s in samples], 0.0)
        batch.update({
            'tokens': tokens,
            'mel2ph': mel2ph,
            'mel': mel,
            'f0': f0,
        })
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad)
        if self.need_key_shift:
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])[:, None]
        if self.need_speed:
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])[:, None]
        if self.need_spk_id:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if self.need_lang_id:
            languages = utils.collate_nd([s['languages'] for s in samples], 0)
            batch['languages'] = languages
        if self.use_alf:
            # needs_alignment flag and actual mel lengths (needed for ALF prior computation)
            batch['needs_alignment'] = torch.BoolTensor(
                [bool(s.get('needs_alignment', False)) for s in samples]
            )
            batch['mel_lengths'] = torch.LongTensor([s['mel'].shape[0] for s in samples])
        return batch


class AcousticTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = AcousticDataset
        self.diffusion_type = hparams['diffusion_type']
        assert self.diffusion_type in ['ddpm', 'reflow'], f"Unknown diffusion type: {self.diffusion_type}"
        self.use_shallow_diffusion = hparams['use_shallow_diffusion']
        if self.use_shallow_diffusion:
            self.shallow_args = hparams['shallow_diffusion_args']
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']

        self.use_alf = hparams.get('use_alf', False)

        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.logged_gt_wav = set()
        self.required_variances = []
        if hparams['use_energy_embed']:
            self.required_variances.append('energy')
        if hparams['use_breathiness_embed']:
            self.required_variances.append('breathiness')
        if hparams['use_voicing_embed']:
            self.required_variances.append('voicing')
        if hparams['use_tension_embed']:
            self.required_variances.append('tension')
        super()._finish_init()

    def _build_model(self):
        return DiffSingerAcoustic(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.use_shallow_diffusion:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']
            self.register_validation_loss('aux_mel_loss')
        if self.diffusion_type == 'ddpm':
            self.mel_loss = DiffusionLoss(loss_type=hparams['main_loss_type'])
        elif self.diffusion_type == 'reflow':
            self.mel_loss = RectifiedFlowLoss(
                loss_type=hparams['main_loss_type'], log_norm=hparams['main_loss_log_norm']
            )
        else:
            raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
        self.register_validation_loss('mel_loss')
        if self.use_alf:
            from modules.alignment import AttentionCTCLoss, AttentionBinarizationLoss
            self.alf_ctc_loss = AttentionCTCLoss()
            self.alf_bin_loss = AttentionBinarizationLoss()
            self.lambda_alf_ctc = hparams.get('lambda_alf_ctc', 1.0)
            self.lambda_alf_bin = hparams.get('lambda_alf_bin', 1.0)
            self.register_validation_loss('alf_ctc_loss')
            self.register_validation_loss('alf_bin_loss')

    def run_model(self, sample, infer=False, return_output=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        target = sample['mel']  # [B, T_s, M]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')

        if hparams['use_spk_id']:
            spk_embed_id = sample['spk_ids']
        else:
            spk_embed_id = None
        if hparams['use_lang_id']:
            languages = sample['languages']
        else:
            languages = None

        needs_alignment = sample.get('needs_alignment') if self.use_alf else None
        mel_lengths = sample.get('mel_lengths') if self.use_alf else None

        output: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph, f0=f0, **variances,
            key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, languages=languages,
            gt_mel=target, infer=infer,
            needs_alignment=needs_alignment,
            mel_lengths=mel_lengths,
        )

        if infer:
            return output
        else:
            losses = {}

            if output.aux_out is not None:
                aux_out = output.aux_out
                norm_gt = self.model.aux_decoder.norm_spec(target)
                aux_mel_loss = self.lambda_aux_mel_loss * self.aux_mel_loss(aux_out, norm_gt)
                losses['aux_mel_loss'] = aux_mel_loss

            # For mel loss, use ALF-derived mel2ph non-padding mask when available
            if output.alf_out is not None:
                _, _, _, token_lengths, mel_lens = output.alf_out
                # Build non-padding mask from mel_lengths (ALF items have correct mel_lens)
                B, T_mel = mel2ph.shape
                mel_idx = torch.arange(T_mel, device=mel2ph.device)[None]  # [1, T_mel]
                if needs_alignment is not None and needs_alignment.any():
                    alf_mask = mel_idx < mel_lens[:, None]  # [B, T_mel] True for valid frames
                    gt_mask = mel2ph > 0
                    combined_mask = torch.where(needs_alignment[:, None], alf_mask, gt_mask)
                    non_padding = combined_mask.unsqueeze(-1).float()
                else:
                    non_padding = (mel2ph > 0).unsqueeze(-1).float()
            else:
                non_padding = (mel2ph > 0).unsqueeze(-1).float()

            if output.diff_out is not None:
                if self.diffusion_type == 'ddpm':
                    x_recon, x_noise = output.diff_out
                    mel_loss = self.mel_loss(x_recon, x_noise, non_padding=non_padding)
                elif self.diffusion_type == 'reflow':
                    v_pred, v_gt, t = output.diff_out
                    mel_loss = self.mel_loss(v_pred, v_gt, t=t, non_padding=non_padding)
                else:
                    raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
                losses['mel_loss'] = mel_loss

            # ALF losses (only when any item in the batch needed alignment)
            if output.alf_out is not None and needs_alignment is not None and needs_alignment.any():
                attn_softs, attn_hards, attn_logprobs, token_lengths, mel_lens = output.alf_out
                alf_ctc = self.lambda_alf_ctc * self.alf_ctc_loss(
                    attn_logprobs, token_lengths, mel_lens
                )
                alf_bin = self.lambda_alf_bin * self.alf_bin_loss(attn_hards, attn_softs)
                losses['alf_ctc_loss'] = alf_ctc
                losses['alf_bin_loss'] = alf_bin

            if return_output:
                return losses, output
            return losses

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _validation_step(self, sample, batch_idx):
        losses, non_infer_output = self.run_model(sample, infer=False, return_output=True)
        if sample['size'] > 0 and min(sample['indices']) < hparams['num_valid_plots']:
            mel_out: ShallowDiffusionOutput = self.run_model(sample, infer=True)
            for i in range(len(sample['indices'])):
                data_idx = sample['indices'][i].item()
                if data_idx < hparams['num_valid_plots']:
                    if self.use_vocoder:
                        self.plot_wav(
                            data_idx, sample['mel'][i],
                            mel_out.aux_out[i] if mel_out.aux_out is not None else None,
                            mel_out.diff_out[i],
                            sample['f0'][i]
                        )
                    if mel_out.aux_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.aux_out[i], 'auxmel')
                    if mel_out.diff_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.diff_out[i], 'diffmel')
            # Visualize ALF alignments from the non-infer pass
            if self.use_alf and non_infer_output.alf_out is not None:
                attn_softs, attn_hards, _, token_lengths, mel_lengths = non_infer_output.alf_out
                needs_alignment = sample.get('needs_alignment')
                for i in range(len(sample['indices'])):
                    data_idx = sample['indices'][i].item()
                    if data_idx < hparams['num_valid_plots']:
                        if needs_alignment is None or needs_alignment[i]:
                            t_len = token_lengths[i].item()
                            m_len = mel_lengths[i].item()
                            soft = attn_softs[i, 0, :m_len, :t_len]
                            hard = attn_hards[i, 0, :m_len, :t_len]
                            ph_seq = None
                            if 'ph_texts' in self.valid_dataset.metadata:
                                ph_seq = self.valid_dataset.metadata['ph_texts'][data_idx].split()
                            self.plot_alf(data_idx, soft, hard, ph_seq=ph_seq)
        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_wav(self, data_idx, gt_mel, aux_mel, diff_mel, f0):
        f0_len = self.valid_dataset.metadata['f0'][data_idx]
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        gt_mel = gt_mel[:mel_len].unsqueeze(0)
        if aux_mel is not None:
            aux_mel = aux_mel[:mel_len].unsqueeze(0)
        if diff_mel is not None:
            diff_mel = diff_mel[:mel_len].unsqueeze(0)
        f0 = f0[:f0_len].unsqueeze(0)
        if data_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav_torch(gt_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'gt_{data_idx}', gt_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )
            self.logged_gt_wav.add(data_idx)
        if aux_mel is not None:
            aux_wav = self.vocoder.spec2wav_torch(aux_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'aux_{data_idx}', aux_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )
        if diff_mel is not None:
            diff_wav = self.vocoder.spec2wav_torch(diff_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'diff_{data_idx}', diff_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )

    def plot_mel(self, data_idx, gt_spec, out_spec, name_prefix='mel'):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        spec_cat = torch.cat([(out_spec - gt_spec).abs() + vmin, gt_spec, out_spec], -1)
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'{name_prefix}_{data_idx}', spec_to_figure(
            spec_cat[:mel_len], vmin, vmax, title_text
        ), global_step=self.global_step)

    def plot_alf(self, data_idx, attn_soft, attn_hard, ph_seq=None):
        if isinstance(attn_soft, torch.Tensor):
            attn_soft = attn_soft.detach().cpu().numpy()
        if isinstance(attn_hard, torch.Tensor):
            attn_hard = attn_hard.detach().cpu().numpy()
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(
            f'alf_{data_idx}',
            alf_to_figure(attn_soft, attn_hard, ph_seq=ph_seq, title=title_text),
            global_step=self.global_step
        )
