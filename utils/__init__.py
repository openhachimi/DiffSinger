from __future__ import annotations

import pathlib
import re
from collections import OrderedDict

def _extract_checkpoint_step(ckpt_name: str):
    matched = re.search(r'(?:steps[=_])(\d+)', ckpt_name)
    return None if matched is None else int(matched.group(1))


def _is_checkpoint_file(ckpt_name: str):
    return (
        re.fullmatch(r'model_ckpt_steps_\d+\.ckpt', ckpt_name) is not None or
        re.fullmatch(r'model-.*-steps=\d+-epochs=.*\.ckpt', ckpt_name) is not None
    )


def load_ckpt(
        cur_model, ckpt_base_dir, ckpt_steps=None,
        prefix_in_ckpt='model', ignored_prefixes=None, key_in_ckpt='state_dict',
        strict=True, device='cpu'
):
    import torch

    if ignored_prefixes is None:
        # NOTICE: this is for compatibility with old checkpoints which have duplicate txt_embed layer in them.
        ignored_prefixes = ['model.fs2.encoder.embed_tokens']
    if not isinstance(ckpt_base_dir, pathlib.Path):
        ckpt_base_dir = pathlib.Path(ckpt_base_dir)
    if ckpt_base_dir.is_file():
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in ckpt_base_dir.iterdir()
                if ckpt_file.is_file()
                and _is_checkpoint_file(ckpt_file.name)
                and _extract_checkpoint_step(ckpt_file.name) == int(ckpt_steps)
            ],
            key=lambda x: x.name
        )
        if len(checkpoint_path) == 0:
            checkpoint_path = [ckpt_base_dir / f'model_ckpt_steps_{int(ckpt_steps)}.ckpt']
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in base_dir.iterdir()
                if ckpt_file.is_file() and _is_checkpoint_file(ckpt_file.name)
            ],
            key=lambda x: (_extract_checkpoint_step(x.name), x.name)
        )
    assert len(checkpoint_path) > 0, f'| ckpt not found in {ckpt_base_dir}.'
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(cur_model, CategorizedModule):
        cur_model.check_category(ckpt_loaded.get('category'))
    if key_in_ckpt is None:
        state_dict = ckpt_loaded
    else:
        state_dict = ckpt_loaded[key_in_ckpt]
    if prefix_in_ckpt is not None:
        state_dict = OrderedDict({
            k[len(prefix_in_ckpt) + 1:]: v
            for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
            if all(not k.startswith(p) for p in ignored_prefixes)
        })
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    cur_model.load_state_dict(state_dict, strict=strict)
    shown_model_name = 'state dict'
    if prefix_in_ckpt is not None:
        shown_model_name = f'\'{prefix_in_ckpt}\''
    elif key_in_ckpt is not None:
        shown_model_name = f'\'{key_in_ckpt}\''
    print(f'| load {shown_model_name} from \'{checkpoint_path}\'.')


def remove_suffix(string: str, suffix: str):
    #  Just for Python 3.8 compatibility, since `str.removesuffix()` API of is available since Python 3.9
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string
