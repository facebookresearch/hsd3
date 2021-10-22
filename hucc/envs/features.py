# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import gym
import numpy as np
from dm_control import mujoco
from scipy.spatial.transform import Rotation

from bisk.features import Featurizer
from bisk.features.joints import JointsRelZFeaturizer


class BodyFeetWalkerFeaturizer(Featurizer):
    def __init__(
        self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        assert robot == 'walker', f'Walker robot expected, got "{robot}"'
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        root = self.p.named.data.qpos[[f'{self.prefix}/root{p}' for p in 'zxy']]
        torso_frame = self.p.named.data.xmat[f'{self.prefix}/torso'].reshape(
            3, 3
        )
        torso_pos = self.p.named.data.xpos[f'{self.prefix}/torso']
        positions = []
        for side in ('left', 'right'):
            torso_to_limb = (
                self.p.named.data.xpos[f'{self.prefix}/{side}_foot'] - torso_pos
            )
            # We're in 2D effectively, y is constant
            positions.append(torso_to_limb.dot(torso_frame)[[0, 2]])
        extremities = np.hstack(positions)
        return np.concatenate([root, extremities])

    def feature_names(self) -> List[str]:
        names = ['rootz:p', 'rootx:p', 'rooty:p']
        names += [f'left_foot:p{p}' for p in 'xz']
        names += [f'right_foot:p{p}' for p in 'xz']
        return names


class BodyFeetRelZWalkerFeaturizer(BodyFeetWalkerFeaturizer):
    def __init__(
        self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[0] = self.relz()
        return obs


class BodyFeetHumanoidFeaturizer(Featurizer):
    def __init__(
        self,
        p: mujoco.Physics,
        robot: str,
        prefix: str = 'robot',
        exclude: str = None,
    ):
        super().__init__(p, robot, prefix, exclude)
        self.for_pos = None
        self.for_twist = None
        self.foot_anchor = 'pelvis'
        self.reference = 'torso'
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    @staticmethod
    def decompose_twist_swing_z(q):
        p = [0.0, 0.0, q[2]]
        twist = Rotation.from_quat(np.array([p[0], p[1], p[2], q[3]]))
        swing = Rotation.from_quat(q) * twist.inv()
        return twist, swing

    def __call__(self) -> np.ndarray:
        root = self.p.data.qpos[0:3]
        if self.for_pos is not None:
            root = root.copy()
            root[0:2] -= self.for_pos
            root[0:2] = self.for_twist.apply(root * np.array([1, 1, 0]))[0:2]
        q = self.p.data.qpos[3:7]
        t, s = self.decompose_twist_swing_z(q[[1, 2, 3, 0]])
        tz = t.as_rotvec()[2]
        e = s.as_euler('yzx')
        sy, sx = e[0], e[2]

        # Feet positions are relative to pelvis position and its heading
        # Also, exclude hands for now.
        pelvis_q = self.p.named.data.xquat[f'{self.prefix}/{self.foot_anchor}']
        pelvis_t, pelvis_s = self.decompose_twist_swing_z(
            pelvis_q[[1, 2, 3, 0]]
        )
        pelvis_pos = self.p.named.data.xpos[f'{self.prefix}/{self.foot_anchor}']
        positions = []
        for ex in ('foot',):
            for side in ('left', 'right'):
                pelvis_to_limb = (
                    self.p.named.data.xpos[f'{self.prefix}/{side}_{ex}']
                    - pelvis_pos
                )
                positions.append(pelvis_t.apply(pelvis_to_limb))
        extremities = np.hstack(positions)
        return np.concatenate([root, np.asarray([tz, sy, sx]), extremities])

    def feature_names(self) -> List[str]:
        names = [f'root:p{f}' for f in 'xyz']
        names += [f'root:t{f}' for f in 'z']
        names += [f'root:s{f}' for f in 'yx']
        names += [f'left_foot:p{f}' for f in 'xyz']
        names += [f'right_foot:p{f}' for f in 'xyz']
        return names


class BodyFeetRelZHumanoidFeaturizer(BodyFeetHumanoidFeaturizer):
    def __init__(
        self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[2] = self.relz()
        return obs


def bodyfeet_featurizer(
    p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs
):
    if robot == 'walker':
        return BodyFeetWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No bodyfeet featurizer for robot "{robot}"')


def bodyfeet_relz_featurizer(
    p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs
):
    if robot == 'walker':
        return BodyFeetRelZWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetRelZHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No bodyfeet-relz featurizer for robot "{robot}"')
