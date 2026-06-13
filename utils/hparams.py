from __future__ import annotations

import copy
import pathlib

hparams = {}


def _to_plain_dict(config):
    if config is None:
        return {}
    if hasattr(config, 'model_dump'):
        return config.model_dump(mode='python')
    return copy.deepcopy(config)


def set_hparams(
        config_path: str | None = None,
        config: dict | None = None,
        work_dir: str | pathlib.Path | None = None,
        ckpt_path: str | pathlib.Path | None = None,
        print_hparams: bool = False
):
    if config is None:
        if config_path is None:
            raise ValueError('Either config or config_path must be provided.')
        from lib.config.io import load_raw_config
        config = load_raw_config(pathlib.Path(config_path))
    config = _to_plain_dict(config)
    from lib.config.schema import ConfigurationScope, InferenceConfig, ModelConfig
    if config['model'].get('spec_decoder') is not None or config.get('inference', {}).get('vocoder') is not None:
        scope = ConfigurationScope.ACOUSTIC
    else:
        scope = ConfigurationScope.VARIANCE

    model_config = ModelConfig.model_validate(config['model'], scope=scope)
    inference_config = InferenceConfig.model_validate(config.get('inference', {}), scope=scope)

    if work_dir is None:
        if config_path is not None:
            work_dir = pathlib.Path(config_path).resolve().parent
        else:
            raise ValueError('work_dir must be provided when config_path is omitted.')
    work_dir = pathlib.Path(work_dir).resolve()
    ckpt_path = None if ckpt_path is None else pathlib.Path(ckpt_path).resolve()

    features = config.get('binarizer', {}).get('features', {})
    midi = config.get('binarizer', {}).get('midi', {})
    dictionaries = config.get('data', {}).get('dictionaries')
    glide_types = config.get('data', {}).get('glide_tags', ['up', 'down'])

    vocoder_config = inference_config.vocoder
    if vocoder_config is not None:
        audio_sample_rate = vocoder_config.audio_sample_rate
        hop_size = vocoder_config.hop_size
        fft_size = vocoder_config.fft_size
        win_size = vocoder_config.win_size
        fmin = vocoder_config.spectrogram.fmin
        fmax = vocoder_config.spectrogram.fmax
        audio_num_mel_bins = vocoder_config.spectrogram.num_bins
        timestep = hop_size / audio_sample_rate
    else:
        audio_sample_rate = features.get('audio_sample_rate')
        hop_size = features.get('hop_size')
        fft_size = features.get('fft_size')
        win_size = features.get('win_size')
        spectrogram = features.get('spectrogram', {})
        fmin = spectrogram.get('fmin')
        fmax = spectrogram.get('fmax')
        audio_num_mel_bins = spectrogram.get('num_bins')
        timestep = inference_config.timestep

    augmentation_args = {
        'random_pitch_shifting': {
            'range': list(inference_config.key_shift_range or [-5., 5.])
        },
        'random_time_stretching': {
            'range': list(inference_config.speed_range or [0.5, 2.0])
        }
    }

    values = {
        'work_dir': work_dir,
        'checkpoint_path': ckpt_path,
        'model_config': model_config,
        'inference_config': inference_config,
        'vocoder_config': vocoder_config,
        'audio_sample_rate': audio_sample_rate,
        'hop_size': hop_size,
        'fft_size': fft_size,
        'win_size': win_size,
        'fmin': fmin,
        'fmax': fmax,
        'audio_num_mel_bins': audio_num_mel_bins,
        'timestep': timestep,
        'vocoder': None if vocoder_config is None else vocoder_config.vocoder_type,
        'vocoder_ckpt': None if vocoder_config is None else vocoder_config.vocoder_path,
        'use_spk_id': model_config.use_spk_id,
        'use_lang_id': model_config.linguistic_encoder.use_lang_id,
        'use_glide_embed': model_config.melody_encoder is not None and model_config.melody_encoder.use_glide_id,
        'glide_types': glide_types,
        'midi_smooth_width': midi.get('smooth_width', 0.06),
        'predict_pitch': False if model_config.prediction is None else model_config.prediction.predict_pitch,
        'predict_dur': False,
        'use_energy_embed': model_config.embeddings is not None and model_config.embeddings.use_energy_embed,
        'use_breathiness_embed': model_config.embeddings is not None and model_config.embeddings.use_breathiness_embed,
        'use_voicing_embed': model_config.embeddings is not None and model_config.embeddings.use_voicing_embed,
        'use_tension_embed': model_config.embeddings is not None and model_config.embeddings.use_tension_embed,
        'use_key_shift_embed': model_config.embeddings is not None and model_config.embeddings.use_key_shift_embed,
        'use_speed_embed': model_config.embeddings is not None and model_config.embeddings.use_speed_embed,
        'augmentation_args': augmentation_args,
        'dictionaries': dictionaries,
    }
    if model_config.prediction is not None:
        values.update({
            'predict_energy': model_config.prediction.predict_energy,
            'predict_breathiness': model_config.prediction.predict_breathiness,
            'predict_voicing': model_config.prediction.predict_voicing,
            'predict_tension': model_config.prediction.predict_tension,
        })

    hparams.clear()
    hparams.update(values)
    if print_hparams:
        for key in sorted(hparams):
            print(f'{key}: {hparams[key]}')
    return hparams
