import json
import os
import pathlib
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import click

root_dir = Path(__file__).resolve().parent.parent
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams


def _collect_experiment_roots():
    return [root_dir / 'experiments', root_dir / 'checkpoints']


def find_exp(exp):
    exp_path = pathlib.Path(exp)
    if exp_path.exists():
        return exp_path.resolve()
    for base_dir in _collect_experiment_roots():
        candidate = base_dir / exp
        if candidate.exists():
            print(f'| found experiment by name: {candidate}')
            return candidate.resolve()
    for base_dir in _collect_experiment_roots():
        if not base_dir.exists():
            continue
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(exp):
                print(f'| match experiment by prefix: {subdir}')
                return subdir.resolve()
    raise click.BadParameter(
        f'No matching experiment starting with \'{exp}\' was found in '
        f'{", ".join(str(p) for p in _collect_experiment_roots())}.'
    )


def _parse_ckpt(value):
    if value is None:
        return None
    if re.fullmatch(r'\d+', value):
        return int(value)
    ckpt_path = pathlib.Path(value).expanduser()
    if ckpt_path.exists():
        return ckpt_path.resolve()
    raise click.BadParameter('Checkpoint must be either a training step number or an existing .ckpt file path.')


def _checkpoint_step(ckpt_name: str):
    matched = re.search(r'(?:steps[=_])(\d+)', ckpt_name)
    return None if matched is None else int(matched.group(1))


def _find_checkpoint(exp_dir: pathlib.Path, ckpt):
    if isinstance(ckpt, pathlib.Path):
        return ckpt.resolve()
    candidates = sorted(
        [
            ckpt_file for ckpt_file in exp_dir.glob('*.ckpt')
            if _checkpoint_step(ckpt_file.name) is not None
        ],
        key=lambda x: (_checkpoint_step(x.name), x.name)
    )
    if isinstance(ckpt, int):
        for candidate in candidates:
            if _checkpoint_step(candidate.name) == ckpt:
                return candidate.resolve()
        raise click.BadParameter(f'Checkpoint step {ckpt} not found in {exp_dir}.')
    if not candidates:
        raise click.BadParameter(f'No checkpoint file found in {exp_dir}.')
    return candidates[-1].resolve()


def _resolve_runtime_paths(exp, work_dir, config, ckpt):
    if work_dir is not None:
        exp_dir = work_dir.resolve()
    elif exp is not None:
        exp_dir = find_exp(exp)
    elif config is not None:
        exp_dir = config.parent.resolve()
    elif isinstance(ckpt, pathlib.Path):
        exp_dir = ckpt.parent.resolve()
    else:
        raise click.BadParameter('Please provide --exp, --work-dir, --config, or a checkpoint file path via --ckpt.')

    if config is None:
        config = exp_dir / 'config.yaml'
    else:
        config = config.resolve()
    if not config.exists():
        raise click.BadParameter(f'Config file not found: {config}')

    ckpt_path = _find_checkpoint(exp_dir, ckpt)
    return exp_dir, config, ckpt_path


def _load_model_and_inference_config(config_path: pathlib.Path, scope: int):
    from lib.config.io import load_raw_config
    from lib.config.schema import InferenceConfig, ModelConfig

    raw_config = load_raw_config(config_path, inherit=True)
    model_config = ModelConfig.model_validate(raw_config['model'], scope=scope)
    inference_config = InferenceConfig.model_validate(raw_config.get('inference', {}), scope=scope)
    raw_config = {
        'model': model_config.model_dump(mode='python'),
        'inference': inference_config.model_dump(mode='python')
    }
    return raw_config


@click.group()
def main():
    pass


def shared_model_options(func):
    options = [
        click.option('--exp', type=str, required=False, metavar='EXP', help='Experiment name or path.'),
        click.option(
            '--work-dir', type=click.Path(
                exists=True, file_okay=False, dir_okay=True, readable=True,
                path_type=pathlib.Path, resolve_path=True
            ),
            required=False,
            help='Path to the experiment directory containing config.yaml and checkpoints.'
        ),
        click.option(
            '--config', type=click.Path(
                exists=True, file_okay=True, dir_okay=False, readable=True,
                path_type=pathlib.Path, resolve_path=True
            ),
            required=False,
            help='Path to the config file to use for inference.'
        ),
        click.option(
            '--ckpt', type=click.STRING,
            required=False, metavar='STEPS_OR_FILE',
            help='Checkpoint training step number or checkpoint file path.'
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


@main.command(help='Run DiffSinger acoustic model inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@shared_model_options
@click.option('--spk', type=click.STRING, required=False, help='Speaker name or mixture of speakers')
@click.option('--lang', type=click.STRING, required=False, help='Default language name')
@click.option(
    '--out', type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=False, help='Path of the output folder'
)
@click.option('--title', type=click.STRING, required=False, help='Title of output file')
@click.option('--num', type=click.IntRange(min=1), required=False, default=1, help='Number of runs')
@click.option('--key', type=click.INT, required=False, default=0, help='Key transition of pitch')
@click.option('--gender', type=click.FloatRange(min=-1, max=1), required=False, help='Formant shifting (gender control)')
@click.option('--seed', type=click.INT, required=False, default=-1, help='Random seed of the inference')
@click.option('--depth', type=click.FloatRange(min=0, max=1), required=False, help='Shallow diffusion depth')
@click.option('--steps', type=click.IntRange(min=1), required=False, help='Diffusion sampling steps')
@click.option('--mel', is_flag=True, help='Save intermediate mel format instead of waveform')
def acoustic(
        proj: pathlib.Path,
        exp: str,
        work_dir: pathlib.Path,
        config: pathlib.Path,
        ckpt: str,
        spk: str,
        lang: str,
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        gender: float,
        seed: int,
        depth: float,
        steps: int,
        mel: bool
):
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)
    if not isinstance(params, list):
        params = [params]
    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import parse_commandline_spk_mix, trans_key
    from lib.config.schema import ConfigurationScope

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    exp_dir, config_path, ckpt_path = _resolve_runtime_paths(exp, work_dir, config, _parse_ckpt(ckpt))
    config_dict = _load_model_and_inference_config(config_path, scope=ConfigurationScope.ACOUSTIC)
    set_hparams(config=config_dict, work_dir=exp_dir, ckpt_path=ckpt_path)

    if not mel and not pathlib.Path(hparams['vocoder_ckpt']).exists():
        raise click.BadParameter(
            f"Vocoder ckpt '{hparams['vocoder_ckpt']}' not found. "
            f'Please provide a valid vocoder path in the config file.'
        )

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if gender is not None and hparams['use_key_shift_embed']:
            param['gender'] = gender
        if spk_mix is not None:
            param['spk_mix'] = spk_mix
        if lang is not None:
            param['lang'] = lang

    from inference.ds_acoustic import DiffSingerAcousticInfer
    infer_ins = DiffSingerAcousticInfer(load_vocoder=not mel)
    if depth is not None:
        if not infer_ins.model.spec_decoder.use_shallow_diffusion:
            raise click.BadParameter('The selected acoustic model does not use shallow diffusion.')
        min_t_start = infer_ins.model_config.spec_decoder.t_start
        if depth > 1 - min_t_start:
            raise click.BadParameter(f'Depth should not be larger than {1 - min_t_start}.')
        infer_ins.model.spec_decoder.decoder.t_start = 1 - depth
    if steps is not None:
        infer_ins.model.spec_decoder.sampling_steps = steps
    print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name, num_runs=num,
            spk_mix=spk_mix, seed=seed, save_mel=mel
        )
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Run DiffSinger variance model inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@shared_model_options
@click.option('--predict', type=click.STRING, multiple=True, metavar='TAGS', help='Parameters to predict')
@click.option('--spk', type=click.STRING, required=False, help='Speaker name or mixture of speakers')
@click.option('--lang', type=click.STRING, required=False, help='Default language name')
@click.option(
    '--out', type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=False, help='Path of the output folder'
)
@click.option('--title', type=click.STRING, required=False, help='Title of output file')
@click.option('--num', type=click.IntRange(min=1), required=False, default=1, help='Number of runs')
@click.option('--key', type=click.INT, required=False, default=0, help='Key transition of pitch')
@click.option('--expr', type=click.FloatRange(min=0, max=1), required=False, help='Static expressiveness control')
@click.option('--seed', type=click.INT, required=False, default=-1, help='Random seed of the inference')
@click.option('--steps', type=click.IntRange(min=1), required=False, help='Diffusion sampling steps')
def variance(
        proj: pathlib.Path,
        exp: str,
        work_dir: pathlib.Path,
        config: pathlib.Path,
        ckpt: str,
        spk: str,
        lang: str,
        predict: Tuple[str],
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        expr: float,
        seed: int,
        steps: int
):
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent
    if out.resolve() == proj.parent.resolve() and not title:
        name += '_variance'

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)
    if not isinstance(params, list):
        params = [params]
    params = [OrderedDict(p) for p in params]
    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import parse_commandline_spk_mix, trans_key
    from lib.config.schema import ConfigurationScope

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    exp_dir, config_path, ckpt_path = _resolve_runtime_paths(exp, work_dir, config, _parse_ckpt(ckpt))
    config_dict = _load_model_and_inference_config(config_path, scope=ConfigurationScope.VARIANCE)
    set_hparams(config=config_dict, work_dir=exp_dir, ckpt_path=ckpt_path)

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if expr is not None:
            param['expr'] = expr
        if spk_mix is not None:
            param['spk_mix'] = spk_mix
        if lang is not None:
            param['lang'] = lang

    from inference.ds_variance import DiffSingerVarianceInfer
    infer_ins = DiffSingerVarianceInfer(predictions=set(predict))
    if steps is not None:
        if infer_ins.model.predict_pitch:
            infer_ins.model.pitch_predictor.sampling_steps = steps
        if infer_ins.model.predict_variances:
            infer_ins.model.variance_predictor.sampling_steps = steps
    print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(params, out_dir=out, title=name, num_runs=num, seed=seed)
    except KeyboardInterrupt:
        exit(-1)


if __name__ == '__main__':
    main()
