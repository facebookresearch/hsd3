# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn
from torch.jit import trace


class TracedModule(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self._traced = None

    def forward(self, inp):
        if self._traced is None:
            self._traced = trace(self.m, inp)
        return self._traced(inp)
