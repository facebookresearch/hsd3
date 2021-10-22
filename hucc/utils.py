# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
import signal
from collections import defaultdict
from copy import deepcopy
from types import FrameType, SimpleNamespace
from typing import Dict

import hydra
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer

log = logging.getLogger(__name__)


def make_optim(optim_cfg: DictConfig, model: nn.Module) -> SimpleNamespace:
    def make(ocfg: DictConfig, model: nn.Module):
        params = list(model.parameters())
        if len(params) == 0:
            log.info(
                f'Empty parameter list for module, ignoring optimizer settings'
            )
            return None
        device = params[0].device
        if ocfg.get('fuse', False):
            OmegaConf.set_struct(ocfg, False)
            if device.type == 'cuda' and ocfg['_target_'] == 'torch.optim.Adam':
                try:
                    from apex.optimizers import FusedAdam
                except ImportError:
                    pass
                else:
                    # TODO This should support zero_grad(set_to_none)
                    ocfg['_target_'] = 'apex.optimizers.FusedAdam'
                    log.info('Using apex.optimizers.FusedAdam')
            del ocfg['fuse']
        return hydra.utils.instantiate(ocfg, model.parameters())

    def recurse(ocfg: DictConfig, model: nn.Module):
        if ocfg is None:
            return SimpleNamespace()

        optims: Dict[str, Optimizer] = {}
        for k, v in ocfg.items():
            if '_target_' in v:
                optims[k] = make(v, getattr(model, k))
            else:
                optims[k] = recurse(v, getattr(model, k))
        return SimpleNamespace(**optims)

    return recurse(optim_cfg, model)


def set_checkpoint_fn(fn, *args, **kwargs):
    prev_usr1 = None

    def sigusr1(signum: signal.Signals, frame: FrameType):
        log.info('SIGUSR1 intercepted, calling checkpoint handler')
        fn(*args, **kwargs)
        if (
            prev_usr1 is not None
            and prev_usr1 != signal.SIG_IGN
            and prev_usr1 != signal.SIG_DFL
        ):
            prev_usr1(signum, frame)

    prev_usr1 = signal.signal(signal.SIGUSR1, sigusr1)


# A shorthand for specific gather calls:
# Your input tensor has some extra dimenions that you want to retain. For
# example, input is a BxTxN tensor and you want to select specific time-steps
# via an Bx1 tensor. The only way is to gather(), and that requires you to
# expand the index tensor. Or, `dim_select(input, 1, index)`, and you'll get an
# Bx1xN tensor in return. If the index is just a B tensor, the result will be a
# BxN tensor.
def dim_select(input: th.Tensor, dim: int, index: th.Tensor):
    # TODO this is a bunch of special cases for now... figure out how to
    # generalize it?
    if input.ndim == 2 and index.ndim == 1 and dim == 1:
        return input.gather(1, index.view(-1, 1)).squeeze(1)
    elif input.ndim == 3 and index.ndim == 1 and dim == 0:
        index = index.view(1, -1, 1).expand(1, index.shape[0], input.shape[2])
        return input.gather(0, index).view(-1, input.shape[-1])
    elif input.ndim == 3 and index.ndim == 1 and dim == 1:
        index = index.view(-1, 1, 1).expand(index.shape[0], 1, input.shape[-1])
        return input.gather(1, index).view(-1, input.shape[-1])
    elif input.ndim == 3 and index.ndim == 2 and dim == 1:
        index = index.unsqueeze(-1).expand(*index.shape, input.shape[-1])
        return input.gather(1, index)
    else:
        raise ValueError('Can\'t dim_select this combination of tensors')


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect;
    from https://stackoverflow.com/a/2669120."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def sorted_nicely_sep(l, sep=','):
    by_rank = defaultdict(list)
    for a in l:
        by_rank[len(a.split(sep))].append(a)
    ta_sorted = []
    for k in sorted(by_rank.keys()):
        ta_sorted += sorted_nicely(by_rank[k])
    return ta_sorted
