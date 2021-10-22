# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch as th
from gym.spaces import Box, Dict, Discrete


def th_flatten(space, x) -> th.Tensor:
    '''
    Adapted from gym.spaces.flatten(); accounts for batch dimension.
    '''
    if isinstance(space, Box):
        return x.view(x.shape[0], -1)
    elif isinstance(space, Discrete):
        return x.view(x.shape[0], -1)
    elif isinstance(space, Dict):
        return th.cat(
            [th_flatten(s, x[key]) for key, s in space.spaces.items()], 1
        )
    else:
        raise NotImplementedError()


def th_unflatten(space, x: th.Tensor):
    '''
    Adapted from gym.spaces.unflatten().
    '''
    if isinstance(space, Box):
        return x.view(x.shape[0], *space.shape)
    elif isinstance(space, Discrete):
        return x.view(x.shape[0], *space.shape)
    else:
        raise NotImplementedError()
