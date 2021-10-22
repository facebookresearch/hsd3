# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from gym.envs.registration import register

from bisk.features import register_featurizer
from hucc.envs.features import bodyfeet_featurizer, bodyfeet_relz_featurizer

register(
    id='ContCtrlgsPreTraining-v1',
    entry_point='hucc.envs.ctrlgs:CtrlgsPreTrainingEnv',
    kwargs={
        'reward': 'potential',
        'hard_reset_interval': 100,
        'resample_features': 'soft',
    },
)

register_featurizer('bodyfeet', bodyfeet_featurizer)
register_featurizer('bodyfeet-relz', bodyfeet_relz_featurizer)
