# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch as th
from torch import nn

from bisk import BiskSingleRobotEnv
from bisk.features import make_featurizer
from hucc.envs.goal_spaces import g_goal_spaces, g_delta_feats

log = logging.getLogger(__name__)


class CtrlgsPreTrainingEnv(BiskSingleRobotEnv):
    '''
    A multi-task, goal-based pre-training environment.

    The environment is "empty" except for a single robot that can be controlled.
    The "tasks" consider the control of one or more observed features -- those
    will be sampled according to `feature_dist` (which can also be changed after
    constructing the environment). For each task (combination of features), a
    goal space is constructed using `psi` and `offset`, and goals are sampled in
    this goal space in [-1,1].

    A continual version of this environment can be obtained with a
    `hard_reset_interval` of > 1. This parameter specifices the frequency at
    which the simulation is reset to its initial state. Other resets will simply
    result in a new goal to be sampled.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        feature_dist: Dict[str, float],
        task_map: Dict[str, int],
        precision: float = 0.1,
        idle_steps: int = 0,
        max_steps: int = 20,
        backproject_goal: bool = True,
        reward: str = 'potential',
        hard_reset_interval: int = 1,
        reset_p: float = 0.0,
        resample_features: str = 'hard',
        full_episodes: bool = False,
        allow_fallover: bool = False,
        fallover_penalty: float = -1.0,
        implicit_soft_resets: bool = False,
        goal_sampling: str = 'random',
        ctrl_cost: float = 0.0,
        normalize_gs_observation: bool = False,
        zero_twist_goals: bool = False,
        relative_frame_of_reference: bool = False,
    ):
        # XXX hack to have DMC robots operate with their "native" sensor input
        super().__init__(
            robot=robot,
            features='joints'
            if features not in ('sensorsnoc', 'native')
            else features,
            allow_fallover=allow_fallover,
        )
        self.goal_featurizer = make_featurizer(
            features, self.p, self.robot, 'robot'
        )
        gsdim = self.goal_featurizer.observation_space.shape[0]
        self.goal_space = g_goal_spaces[features][robot]

        # Construct goal space
        self.psi, self.offset = self.abstraction_matrix(robot, features, gsdim)
        self.psi_1 = np.linalg.inv(self.psi)
        self.offset_1 = -np.matmul(self.offset, self.psi_1)

        assert len(self.observation_space.shape) == 1
        assert self.psi.shape == (gsdim, gsdim)
        assert self.offset.shape == (gsdim,)

        self.precision = precision
        self.idle_steps = idle_steps
        self.max_steps = max_steps
        self.backproject_goal = backproject_goal
        self.reward = reward
        self.hard_reset_interval = hard_reset_interval
        self.reset_p = reset_p
        self.resample_features = resample_features
        self.full_episodes = full_episodes
        self.fallover_penalty = fallover_penalty
        self.ctrl_cost = ctrl_cost
        self.implicit_soft_resets = implicit_soft_resets
        self.goal_sampling = goal_sampling
        self.normalize_gs_observation = normalize_gs_observation
        self.zero_twist_goals = zero_twist_goals
        self.relative_frame_of_reference = relative_frame_of_reference

        self.task_idx = [0] * len(task_map)
        for k, v in task_map.items():
            self.task_idx[v] = int(k)

        if len(self.goal_space['twist_feats']) > 0:
            negpi = self.proj(
                -np.pi * np.ones(gsdim), self.goal_space['twist_feats']
            )
            pospi = self.proj(
                np.pi * np.ones(gsdim), self.goal_space['twist_feats']
            )
            if not np.allclose(-negpi, pospi):
                # This could be supported by more elobarte delta computation
                # logic in step()
                raise ValueError('Twist feature ranges not symmetric')
            self.proj_pi = pospi

        if backproject_goal:
            all_feats = list(range(gsdim))
            gmin_back = self.backproj(-np.ones(gsdim), all_feats)
            gmax_back = self.backproj(np.ones(gsdim), all_feats)
            goal_space = gym.spaces.Box(gmin_back, gmax_back)
        else:
            max_features = max(
                (
                    len(f.replace('+', ',').split(','))
                    for f in feature_dist.keys()
                )
            )
            goal_space = gym.spaces.Box(
                low=-2, high=2, shape=(max_features,), dtype=np.float32
            )

        self.task_map = {int(k): v for k, v in task_map.items()}

        # Hide position-related invariant features from the observation, i.e.
        # X/Y or ant X for cheetah
        delta_feats = g_delta_feats[robot]
        self.obs_mask = list(range(self.observation_space.shape[0]))
        for d in delta_feats:
            self.obs_mask.remove(d)

        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.obs_mask),),
                    dtype=np.float32,
                ),
                'desired_goal': goal_space,
                'task': gym.spaces.Box(
                    low=0, high=1, shape=(len(self.task_map),), dtype=np.float32
                ),
                'gs_observation': self.goal_featurizer.observation_space,
            }
        )

        self._do_hard_reset = True
        self._reset_counter = 0
        self.set_feature_dist(feature_dist)
        # Current features
        self._features: List[int] = []
        self._features_s = ''
        self._feature_mask = np.zeros(len(self.task_map))

        self.model = None
        self.gamma = 1.0

    def set_goal_dims(self, dims):
        self.set_feature_dist(dims)

    def set_model(self, model: nn.Module, gamma: float):
        self.model = model
        self.gamma = gamma

    def set_feature_dist(self, feature_dist: Dict[str, float]):
        # Deduplicate features from combinations
        fdist: Dict[str, float] = {}
        self._feature_strings = {}
        for fs, p in feature_dist.items():
            ufeats = []
            for f in fs.replace('+', ',').split(','):
                if not f in ufeats:
                    ufeats.append(f)
            fdist[','.join(ufeats)] = p
            self._feature_strings[','.join(ufeats)] = fs.replace('+', ',')

        if not self.backproject_goal:
            # Check that maximum number of features doesn't change
            max_features = max((len(fs.split(',')) for fs in fdist.keys()))
            assert (
                self.observation_space['desired_goal'].shape[0] == max_features
            )
        for fs in fdist.keys():
            for fi in map(int, fs.split(',')):
                assert fi in self.task_map
        self._feature_dist_v = [k for k, v in fdist.items()]
        s = sum([v for k, v in fdist.items()])
        self._feature_dist_p = [v / s for k, v in fdist.items()]

    def proj(self, obs: np.ndarray, feats: List[int]) -> np.ndarray:
        return np.matmul(obs, self.psi[feats].T) + self.offset[feats]

    def backproj(self, obs_w: np.ndarray, feats: List[int]) -> np.ndarray:
        s_p = np.matmul(obs_w, self.psi_1[feats]) + self.offset_1
        return s_p[self.task_idx]

    def seed(self, seed=None):
        self._do_hard_reset = True
        return super().seed(seed)

    def get_observation(self):
        obs = super().get_observation()[self.obs_mask]
        gs_obs = self.goal_featurizer()
        if self.backproject_goal:
            s = gs_obs[self.task_idx]
            bpg = self.backproj(self.goal, self._features)
            g = bpg - s
            if len(self.goal_space['twist_feats']) > 0:
                twf = [self.task_map[f] for f in self.goal_space['twist_feats']]
                g[twf] = (
                    np.remainder((bpg[twf] - s[twf]) + np.pi, 2 * np.pi) - np.pi
                )
            g *= self._feature_mask
        else:
            if len(self.goal_space['twist_feats']) > 0:
                raise NotImplementedError()
            gs = self.proj(gs_obs, self._features)
            g = np.zeros(self.observation_space['desired_goal'].shape)
            g[0 : len(self.goal)] = self.goal - gs
        if self.normalize_gs_observation:
            # XXX if the goal space is defined for fewer features than
            # gs_observation, this will be yield bogus values for undefined
            # ones.
            gs_obs = self.proj(gs_obs, np.arange(0, len(gs_obs)))
        return {
            'observation': obs,
            'desired_goal': g,
            'task': self._feature_mask,
            'gs_observation': gs_obs,
        }

    def hard_reset(self):
        # Disable contacts during reset to prevent potentially large contact
        # forces that can be applied during initial positioning of bodies in
        # reset_state().
        with self.p.model.disable('contact'):
            self.p.reset()
            self.reset_state()

        for _ in range(self.idle_steps):
            self.p.set_control(np.zeros_like(self.p.data.ctrl))
            self.step_simulation()
        if self.idle_steps <= 0:
            self.step_simulation()

    def sample_features(self) -> List[int]:
        fs = self.np_random.choice(
            self._feature_dist_v, 1, p=self._feature_dist_p
        )[0]
        return list(map(int, fs.split(',')))

    def sample_goals_random(self, N: int = 1) -> np.ndarray:
        gstate = self.proj(self.goal_featurizer(), self._features)
        goal = self.np_random.uniform(
            low=-1.0, high=1.0, size=(N, len(self._features))
        )
        # For delta features we offset the goal by the current state to get
        # meaningful deltas afterwards
        for i, f in enumerate(self._features):
            if f in self.goal_space['delta_feats']:
                goal[:, i] += gstate[i]
            if self.zero_twist_goals and f in self.goal_space['twist_feats']:
                goal[:, i] = 0
        return goal

    def sample_goal_using_r(self) -> np.ndarray:
        N = 128
        cand = self.sample_goals_random(N=N)
        if self.backproject_goal:
            s = self.goal_featurizer()[self.task_idx]
            gb = (np.matmul(cand, self.psi_1[self._features]) + self.offset_1)[
                :, self.task_idx
            ]
            g = gb - s
            g *= self._feature_mask
        else:
            gs = self.proj(self.goal_featurizer(), self._features)
            g = np.zeros((N, self.observation_space['desired_goal'].shape[0]))
            g[:, 0 : len(self._features)] = cand - gs

        obs = super().get_observation()[self.obs_mask]
        inp = {
            'observation': th.tensor(obs, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, obs.shape[0]),
            'desired_goal': th.tensor(g, dtype=th.float32),
            'task': th.tensor(self._feature_mask, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, self._feature_mask.shape[0]),
        }
        with th.no_grad():
            action = self.model.pi(inp).mean
        inp['action'] = action
        with th.no_grad():
            r = self.model.reachability(inp).clamp(0, 1)
        if self.goal_sampling in {'r2', 'reachability2'}:
            # Favor samples reachable with 50% probability
            dist = th.tanh(2 * (1 - th.abs(r * 2 - 1) + 1e-1))
        else:
            # Favor unreachable samples
            dist = 1 / (r.view(-1) + 0.1)
        return cand[th.multinomial(dist, 1).item()]

    def sample_goal_using_q(self, obs: np.ndarray) -> np.ndarray:
        N = 128
        cand = self.sample_goals_random(N=N)
        if self.backproject_goal:
            s = self.goal_featurizer()[self.task_idx]
            gb = (np.matmul(cand, self.psi_1[self._features]) + self.offset_1)[
                :, self.task_idx
            ]
            g = gb - s
            g *= self._feature_mask
        else:
            gs = self.proj(self.goal_featurizer(), self._features)
            g = np.zeros((N, self.observation_space['desired_goal'].shape[0]))
            g[:, 0 : len(self._features)] = cand - gs

        obs = super().get_observation()[self.obs_mask]
        inp = {
            'observation': th.tensor(obs, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, obs.shape[0]),
            'desired_goal': th.tensor(g, dtype=th.float32),
            'task': th.tensor(self._feature_mask, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, self._feature_mask.shape[0]),
        }
        with th.no_grad():
            action = self.model.pi(inp).mean
        inp['action'] = action
        with th.no_grad():
            q = th.min(self.model.q(inp), dim=-1).values

        ctrl_cost = (
            self.max_steps
            * self.ctrl_cost
            * (0.25 * self.action_space.shape[0])
        )
        wobs = self.proj(obs, self._features)
        dist = np.linalg.norm(cand - wobs, ord=2, axis=1)
        min_ret = (dist - ctrl_cost) * self.gamma ** self.max_steps
        slack = q - min_ret
        dist = 1 / (slack - slack.min() + 1)
        return cand[th.multinomial(dist, 1).item()]

    def reset(self):
        need_hard_reset = self._do_hard_reset or (
            self.hard_reset_interval > 0
            and self._reset_counter % self.hard_reset_interval == 0
        )
        # Reset
        if need_hard_reset:
            self.hard_reset()
            self._reset_counter = 0
        if self.relative_frame_of_reference:
            self.goal_featurizer.set_frame_of_reference()

        # Sample features and goal
        resample_features = False
        if need_hard_reset:
            resample_features = True
        if self.resample_features == 'soft':
            resample_features = True
        elif self.resample_features.startswith('soft'):
            freq = int(self.resample_features[4:])
            resample_features = self._reset_counter % freq == 0
        if resample_features:
            self._features = self.sample_features()
            self._features_s = self._feature_strings[
                ','.join(map(str, self._features))
            ]
            self._feature_mask *= 0
            for f in self._features:
                self._feature_mask[self.task_map[f]] = 1.0

        self.goal = self.sample_goals_random()[0]
        if self.goal_sampling in {'q', 'q_value'}:
            if self.model:
                self.goal = self.sample_goal_using_q()
        elif self.goal_sampling in {'r', 'reachability', 'r2', 'reachability2'}:
            if self.model:
                self.goal = self.sample_goal_using_r()
        elif self.goal_sampling not in {'random', 'uniform'}:
            raise ValueError(
                f'Unknown goal sampling method "{self.goal_sampling}"'
            )

        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space['twist_feats']:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        self._d_initial = distance_to_goal()

        self._do_hard_reset = False
        self._reset_counter += 1
        self._step = 0
        return self.get_observation()

    def step(self, action):
        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space['twist_feats']:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        d_prev = distance_to_goal()
        next_obs, reward, done, info = super().step(action)
        d_new = distance_to_goal()

        info['potential'] = d_prev - d_new
        info['distance'] = d_new
        info['reached_goal'] = info['distance'] < self.precision
        if self.reward == 'potential':
            reward = info['potential']
        elif self.reward == 'potential2':
            reward = d_prev - self.gamma * d_new
        elif self.reward == 'potential3':
            reward = 1.0 if info['reached_goal'] else 0.0
            reward += d_prev - self.gamma * d_new
        elif self.reward == 'potential4':
            reward = (d_prev - d_new) / self._d_initial
        elif self.reward == 'distance':
            reward = -info['distance']
        elif self.reward == 'sparse':
            reward = 1.0 if info['reached_goal'] else 0.0
        else:
            raise ValueError(f'Unknown reward: {self.reward}')
        reward -= self.ctrl_cost * np.square(action).sum()

        info['EpisodeContinues'] = True
        if info['reached_goal'] == True and not self.full_episodes:
            done = True
        info['time'] = self._step
        self._step += 1
        if self._step >= self.max_steps:
            done = True
        elif (
            not info['reached_goal'] and self.np_random.random() < self.reset_p
        ):
            info['RandomReset'] = True
            done = True

        if not self.allow_fallover and self.fell_over():
            reward = self.fallover_penalty
            done = True
            self._do_hard_reset = True
            info['reached_goal'] = False
            info['fell_over'] = True
        if done and (
            self._do_hard_reset
            or (self._reset_counter % self.hard_reset_interval == 0)
        ):
            del info['EpisodeContinues']
        if done:
            info['LastStepOfTask'] = True

        if done and 'EpisodeContinues' in info and self.implicit_soft_resets:
            need_hard_reset = self._do_hard_reset or (
                self.hard_reset_interval > 0
                and self._reset_counter % self.hard_reset_interval == 0
            )
            if not need_hard_reset:
                # Do implicit resets, let episode continue
                next_obs = self.reset()
                done = False
                del info['EpisodeContinues']
                info['SoftReset'] = True

        info['features'] = self._features_s
        return next_obs, reward, done, info

    @staticmethod
    def feature_controllable(robot: str, features: str, dim: int) -> bool:
        if not features in g_goal_spaces:
            raise ValueError(f'Unsupported feature space: {robot}')
        if not robot in g_goal_spaces[features]:
            raise ValueError(f'Unsupported robot: {robot}')
        gs = g_goal_spaces[features][robot]
        if dim < 0:
            raise ValueError(f'Feature {dim} out of range')
        if dim >= len(gs['min']):
            return False
        # Return whether feature is controllable, i.e. range is non-zero
        return gs['min'][dim] != gs['max'][dim]

    @staticmethod
    def abstraction_matrix(
        robot: str, features: str, sdim: int
    ) -> Tuple[np.array, np.array]:
        if not features in g_goal_spaces:
            raise ValueError(f'Unsupported feature space: {robot}')
        if not robot in g_goal_spaces[features]:
            raise ValueError(f'Unsupported robot: {robot}')
        gs = g_goal_spaces[features][robot]
        gmin = np.array(gs['min'])
        gmax = np.array(gs['max'])
        if gmin.size == 0:
            # Dummy values
            gmin = -np.ones(sdim)
            gmax = np.ones(sdim)
        if len(gmin) < sdim:
            gmin = np.concatenate([gmin, np.zeros(sdim - len(gmin))])
            gmax = np.concatenate([gmax, np.zeros(sdim - len(gmax))])
        psi = 2 * (np.eye(len(gmin)) * 1 / (gmax - gmin + 1e-7))
        offset = -2 * (gmin / (gmax - gmin + 1e-7)) - 1
        return psi, offset

    def delta_features(robot: str, features: str) -> List[int]:
        return g_goal_spaces[features][robot]['delta_feats']

    def feature_name(robot: str, features: str, f: int) -> str:
        return g_goal_spaces[features][robot]['str'][f]
