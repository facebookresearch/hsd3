# Copyright (c) 2020-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import numpy as np
import torch as th


class HashingCountReward:
    '''
    Hash-based count bonus for exploration with SimHash.
    Adapted from https://github.com/openai/EPG/blob/bb73d77/epg/exploration.py
    (MIT license) and
    https://github.com/oxwhirl/opiq/blob/b7df513/src/count/HashCount.py (MIT
    License)

    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y.,
    Schulman, J., De Turck, F., and Abbeel, P. (2017).  #Exploration: A study of
    count-based exploration for deep reinforcement learning.  In Advances in
    Neural Information Processing Systems (NIPS).
    '''
    def __init__(self,
                 obs_dim: int,
                 dim_key: int = 32,
                 bucket_sizes: List[int] = None):
        if bucket_sizes is None:
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = th.tensor(bucket_sizes, dtype=th.long)
        self.mods_list = th.tensor(mods_list, dtype=th.float).transpose(1, 0)
        self.tables = th.zeros((len(bucket_sizes), max(bucket_sizes)))
        self.projection_matrix = th.normal(mean=th.zeros(size=(obs_dim,
                                                               dim_key)),
                                           std=th.ones(size=(obs_dim,
                                                             dim_key)))

    def to_(self, device):
        self.bucket_sizes = self.bucket_sizes.to(device)
        self.mods_list = self.mods_list.to(device)
        self.tables = self.tables.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

    def save(self, dest):
        th.save(self.__dict__, dest)

    def load(self, src):
        d = th.load(src)
        for k in self.__dict__.keys():
            self.__dict__[k] = d[k]

    def compute_keys(self, obss: th.Tensor):
        binaries = th.sign(obss @ self.projection_matrix).float()
        keys = (binaries @ self.mods_list).long() % self.bucket_sizes
        return keys

    def inc_hash_keys(self, keys: th.Tensor):
        self.tables[range(len(self.bucket_sizes)), keys] += 1

    def inc_hash(self, obss: th.Tensor):
        keys = self.compute_keys(obss)
        self.inc_hash_keys(keys)

    def query_hash(self, keys: th.Tensor):
        return self.tables.gather(1, keys.T).min(dim=0).values

    def predict_keys(self, keys: th.Tensor):
        counts = self.query_hash(keys)
        return counts.sqrt().clamp(min=1).reciprocal()

    def predict(self, obss: th.Tensor):
        keys = self.compute_keys(obss)
        return self.predict_keys(keys)
