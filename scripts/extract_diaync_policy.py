# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from typing import Dict

import torch as th

import hucc

inp = sys.argv[1]
outp = sys.argv[2]

m = th.load(inp, map_location='cpu')['_model']
d: Dict[str, th.Tensor] = {}
for k, v in m.items():
    if k.startswith('pi.1'):
        parts = k.split('.')
        parts[2] = str(int(parts[2]) + 1)  # +1 for flattening
        del parts[1]  # remove extra nesting
        dk = '.'.join(parts)
        d[dk] = v
        print(f'{k} -> {dk}')
th.save(d, outp)
