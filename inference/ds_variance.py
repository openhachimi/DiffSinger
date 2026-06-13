import copy
import json
import pathlib
from collections import OrderedDict
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy import interpolate

from basics.base_svs_infer import BaseSVSInfer
from lib.functional import resample_align_curve
from lib.vocabulary import load_phoneme_dictionary
from modules.commons.tts_modules import LengthRegulator
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.toplevel import DiffSingerVariance
from utils import load_ckpt
from utils.hparams import hparams
from lib.feature.pitch import interp_f0


class DiffSingerVarianceInfer(BaseSVSInfer):
    def __init__(self, device=None, ckpt_steps=None, predictions: set = None):
        super().__init__(device=device)
        self.model_config = hparams['model_config']
        self.phoneme_dictionary = load_phoneme_dictionary()
        if hparams['use_spk_id']:
            with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                self.spk_map = json.load(f)
            assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
        lang_map_fn = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
        if lang_map_fn.exists():
            with open(lang_map_fn, 'r', encoding='utf8') as f:
                self.lang_map = json.load(f)
        self.model: DiffSingerVariance = self.build_model(ckpt_steps=ckpt_steps)
        self.lr = LengthRegulator()
        smooth_kernel_size = max(1, round(hparams['midi_smooth_width'] / self.timestep))
        self.smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval().to(self.device)
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ).to(self.device))
        smooth_kernel /= smooth_kernel.sum()
        self.smooth.weight.data = smooth_kernel[None, None]

        glide_types = hparams.get('glide_types', ['up', 'down'])
        assert 'none' not in glide_types, 'Type name \'none\' is reserved and should not appear in glide_types.'
        self.glide_map = {'none': 0, **{typename: idx + 1 for idx, typename in enumerate(glide_types)}}

        predictions = predictions or set()
        self.auto_completion_mode = len(predictions) == 0
        self.global_predict_pitch = 'pitch' in predictions and hparams['predict_pitch']
        self.variance_prediction_set = predictions.intersection(VARIANCE_CHECKLIST)
        self.global_predict_variances = len(self.variance_prediction_set) > 0

    def build_model(self, ckpt_steps=None):
        model = DiffSingerVariance(self.model_config).eval().to(self.device)
        load_ckpt(
            model,
            hparams.get('checkpoint_path') or hparams['work_dir'],
            ckpt_steps=ckpt_steps,
            prefix_in_ckpt='model',
            strict=True,
            device=self.device
        )
        return model

    @torch.no_grad()
    def preprocess_input(self, param, idx=0, load_pitch: bool = False):
        batch = {}
        summary = OrderedDict()

        lang = param.get('lang')
        if lang is None:
            assert len(self.lang_map) <= 1, (
                "This is a multilingual model. "
                "Please specify a language by --lang option."
            )
        else:
            assert lang in self.lang_map, f'Unrecognized language name: \'{lang}\'.'
        if hparams.get('use_lang_id', False):
            languages = torch.LongTensor([
                (
                    self.lang_map[lang if '/' not in p else p.split('/', maxsplit=1)[0]]
                    if self.phoneme_dictionary.is_cross_lingual(p)
                    else 0
                )
                for p in param['ph_seq'].split()
            ]).to(self.device)[None]
            batch['languages'] = languages
        txt_tokens = torch.LongTensor([
            self.phoneme_dictionary.encode(param['ph_seq'], lang=lang)
        ]).to(self.device)
        batch['tokens'] = txt_tokens

        if param.get('ph_dur') is None:
            raise ValueError('v3 variance inference requires ph_dur in the input DS file.')
        ph_dur_sec = torch.from_numpy(np.array([param['ph_dur'].split()], np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
        batch['durations'] = durations
        frame_count = durations.sum().item()

        note_midi = np.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in param['note_seq'].split()],
            dtype=np.float32
        )
        note_rest = note_midi < 0
        if np.all(note_rest):
            note_midi = np.full_like(note_midi, fill_value=60.)
        else:
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            note_midi[note_rest] = interp_func(np.where(note_rest)[0])
        note_midi = torch.from_numpy(note_midi).to(self.device)[None]
        note_rest = torch.from_numpy(note_rest).to(self.device)[None]

        note_dur_sec = torch.from_numpy(np.array([param['note_dur'].split()], np.float32)).to(self.device)
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=1) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=1, prepend=note_acc.new_zeros(1, 1))
        note_frame_count = int(note_dur.sum().item())
        if note_frame_count != frame_count:
            note_dur[:, -1] += frame_count - note_frame_count
        mel2note = self.lr(note_dur)

        summary['notes'] = note_midi.shape[1]
        summary['tokens'] = txt_tokens.shape[1]
        summary['frames'] = frame_count
        summary['seconds'] = '%.2f' % (frame_count * self.timestep)

        if hparams['use_spk_id']:
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='frame', mix_length=frame_count
            )
            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value

        batch['note_midi'] = note_midi
        batch['note_dur'] = note_dur
        batch['note_rest'] = note_rest
        if hparams.get('use_glide_embed', False) and param.get('note_glide') is not None:
            batch['note_glide'] = torch.LongTensor(
                [[self.glide_map.get(x, 0) for x in param['note_glide'].split()]]
            ).to(self.device)
        else:
            batch['note_glide'] = torch.zeros(1, note_midi.shape[1], dtype=torch.long, device=self.device)

        frame_midi_pitch = torch.gather(torch.nn.functional.pad(note_midi, [1, 0]), 1, mel2note)
        batch['base_pitch'] = self.smooth(frame_midi_pitch)

        if load_pitch:
            f0 = resample_align_curve(
                np.array(param['f0_seq'].split(), np.float32),
                original_timestep=float(param['f0_timestep']),
                target_timestep=self.timestep,
                align_length=frame_count
            )
            batch['pitch'] = torch.from_numpy(
                librosa.hz_to_midi(interp_f0(f0)[0]).astype(np.float32)
            ).to(self.device)[None]

        if self.model.predict_pitch:
            if load_pitch:
                summary['pitch'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_pitch:
                summary['pitch'] = 'auto'
            else:
                summary['pitch'] = 'ignored'

        if self.model.predict_variances:
            for v_name in self.model.variance_list:
                if (self.auto_completion_mode and param.get(v_name) is None) or v_name in self.variance_prediction_set:
                    summary[v_name] = 'auto'
                else:
                    summary[v_name] = 'ignored'

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))
        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        if hparams['use_spk_id']:
            spk_mix_id = sample['spk_mix_id']
            spk_mix_value = sample['spk_mix_value']
            spk_mix_embed = torch.sum(
                self.model.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),
                dim=2, keepdim=False
            )
        else:
            spk_mix_embed = None

        pitch_pred, variance_pred, _ = self.model(
            tokens=sample['tokens'],
            durations=sample['durations'],
            languages=sample.get('languages'),
            spk_embed=spk_mix_embed,
            note_midi=sample['note_midi'],
            note_rest=sample['note_rest'],
            note_dur=sample['note_dur'],
            note_glide=sample['note_glide'],
            base_pitch=sample['base_pitch'],
            pitch=sample.get('pitch'),
            infer=True
        )
        return pitch_pred, variance_pred

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            seed: int = -1
    ):
        batches = []
        predictor_flags: List[Tuple[bool, bool]] = []

        for i, param in enumerate(params):
            if self.auto_completion_mode:
                flag = (
                    self.model.predict_pitch and param.get('f0_seq') is None,
                    self.model.predict_variances and any(param.get(v_name) is None for v_name in self.model.variance_list)
                )
            else:
                predict_variances = self.model.predict_variances and self.global_predict_variances
                predict_pitch = self.model.predict_pitch and (
                    self.global_predict_pitch or (param.get('f0_seq') is None and predict_variances)
                )
                flag = (predict_pitch, predict_variances)
            predictor_flags.append(flag)
            batches.append(self.preprocess_input(
                param, idx=i,
                load_pitch=not flag[0] and flag[1]
            ))

        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_runs):
            results = []
            for param, flag, batch in tqdm.tqdm(
                    zip(params, predictor_flags, batches), desc='infer segments', total=len(params)
            ):
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)
                param_copy = copy.deepcopy(param)

                pitch_saved = self.model.predict_pitch
                variance_saved = self.model.predict_variances
                pitch_pred = None
                variance_pred = None
                try:
                    predict_pitch, predict_variances = flag
                    if predict_pitch and predict_variances:
                        self.model.predict_pitch = True
                        self.model.predict_variances = False
                        pitch_pred, _ = self.forward_model(batch)
                        staged_batch = dict(batch)
                        staged_batch['pitch'] = pitch_pred
                        self.model.predict_pitch = False
                        self.model.predict_variances = True
                        _, variance_pred = self.forward_model(staged_batch)
                    elif predict_pitch or predict_variances:
                        self.model.predict_pitch = predict_pitch
                        self.model.predict_variances = predict_variances
                        pitch_pred, variance_pred = self.forward_model(batch)
                finally:
                    self.model.predict_pitch = pitch_saved
                    self.model.predict_variances = variance_saved

                if pitch_pred is not None and (self.auto_completion_mode or self.global_predict_pitch):
                    pitch_pred = pitch_pred[0].cpu().numpy()
                    f0_pred = librosa.midi_to_hz(pitch_pred)
                    param_copy['f0_seq'] = ' '.join([str(round(freq, 1)) for freq in f0_pred.tolist()])
                    param_copy['f0_timestep'] = str(self.timestep)

                variance_outputs = {}
                if variance_pred is not None:
                    variance_outputs = {
                        k: v[0].cpu().numpy()
                        for k, v in zip(self.model.variance_list, variance_pred)
                        if (self.auto_completion_mode and param.get(k) is None) or k in self.variance_prediction_set
                    }
                for v_name, v_pred in variance_outputs.items():
                    param_copy[v_name] = ' '.join([str(round(v, 4)) for v in v_pred.tolist()])
                    param_copy[f'{v_name}_timestep'] = str(self.timestep)

                results.append(param_copy)

            filename = f'{title}-{str(i).zfill(3)}.ds' if num_runs > 1 else f'{title}.ds'
            save_path = out_dir / filename
            with open(save_path, 'w', encoding='utf8') as f:
                print(f'| save params: {save_path}')
                json.dump(results, f, ensure_ascii=False, indent=2)
