# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
import torch.distributions as D
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.envs.ctrlgs import CtrlgsPreTrainingEnv
from hucc.envs.goal_spaces import subsets_task_map
from hucc.models import TracedModule
from hucc.utils import dim_select, sorted_nicely_sep

log = logging.getLogger(__name__)


def _parse_list(s, dtype):
    if s is None:
        return []
    return list(map(dtype, str(s).split('#')))


class DeterministicLo(nn.Module):
    def __init__(self, pi: nn.Module):
        super().__init__()
        self.pi = pi

    def forward(self, x):
        return self.pi(x).mean


class HiToLoInterface:
    '''
    This describes the interface between the high- and low-level sub-agents used
    in HSD3Agent. The goal space handling is a bit involved but needs to stay
    close to the pre-training setup. This interface generally works with a
    number of linear combinations of features; in practice, we manually
    construct these combinatins from a number of specified features.
    '''

    def __init__(self, env, cfg: DictConfig):
        gscfg = cfg.goal_space
        lo_subsets: Optional[List[str]] = None
        lo_task_map: Optional[Dict[str, int]] = None
        try:
            lo_subsets, lo_task_map = self.parse_lo_info(cfg)
        except FileNotFoundError:
            pass

        if gscfg.subsets == 'from_lo':
            subsets, task_map = lo_subsets, lo_task_map
        else:
            subsets, task_map = subsets_task_map(
                features=gscfg.features,
                robot=gscfg.robot,
                spec=gscfg.subsets,
                rank_min=gscfg.rank_min,
                rank_max=gscfg.rank_max,
            )
            if lo_task_map is not None:
                task_map = lo_task_map
        if subsets is None or task_map is None or len(subsets) == 0:
            raise ValueError('No goal space subsets selected')

        self.task_map = task_map
        self.subsets = [s.replace('+', ',') for s in subsets]
        # XXX Unify
        for i in range(len(self.subsets)):
            su = []
            for f in self.subsets[i].split(','):
                if not f in su:
                    su.append(f)
            self.subsets[i] = ','.join(su)
        self.robot = gscfg.robot
        self.features = gscfg.features
        self.delta_actions = bool(gscfg.delta_actions)
        self.mask_gsfeats = _parse_list(gscfg.mask_feats, int)
        n_subsets = len(self.subsets)

        n_obs = env.observation_space['observation'].shape[0]
        self.max_rank = max((len(s.split(',')) for s in self.subsets))
        ng = max(max(map(int, s.split(','))) for s in self.subsets) + 1
        task_space = gym.spaces.Discrete(n_subsets)
        subgoal_space = gym.spaces.Box(
            low=-1, high=1, shape=(ng,), dtype=np.float32
        )
        self.action_space_hi = gym.spaces.Dict(
            [('task', task_space), ('subgoal', subgoal_space)]
        )
        self.action_space_hi.seed(gscfg.seed)
        self.task = th.zeros((n_subsets, len(self.task_map)), dtype=th.float32)
        for i, s in enumerate(self.subsets):
            for j, dim in enumerate(s.split(',')):
                self.task[i][self.task_map[dim]] = 1

        # XXX A very poor way of querying psi etc -- unify this.
        fdist = {a: 1.0 for a in self.subsets}
        dummy_env = CtrlgsPreTrainingEnv(
            gscfg.robot,
            gscfg.features,
            feature_dist=fdist,
            task_map=self.task_map,
        )
        self.gobs_space = dummy_env.observation_space.spaces['gs_observation']
        self.gobs_names = dummy_env.goal_featurizer.feature_names()
        self.goal_space = dummy_env.observation_space.spaces['desired_goal']
        self.delta_feats = dummy_env.goal_space['delta_feats']
        self.twist_feats = [
            self.task_map[str(f)] for f in dummy_env.goal_space['twist_feats']
        ]
        self.psi = dummy_env.psi
        self.offset = dummy_env.offset
        self.psi_1 = dummy_env.psi_1
        self.offset_1 = dummy_env.offset_1
        self.obs_mask = dummy_env.obs_mask
        self.task_idx = dummy_env.task_idx
        gsdim = self.psi.shape[0]
        dummy_env.close()

        self.observation_space_lo = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.obs_mask),),
                    dtype=np.float32,
                ),
                'desired_goal': self.goal_space,
                'task': gym.spaces.Box(
                    low=0, high=1, shape=(len(self.task_map),), dtype=np.float32
                ),
            }
        )
        spaces = copy(env.observation_space.spaces)
        # Ignore the goal space in both policies
        self.gs_key = gscfg.key
        del spaces[self.gs_key]
        self.observation_space_hi = gym.spaces.Dict(spaces)

        # Inverse psi matrix indexed by available subsets
        self.psi_1_by_ss = th.zeros(
            (n_subsets, self.max_rank, gsdim), dtype=th.float32
        )
        self.psi_by_ss = th.zeros(
            (n_subsets, self.max_rank, gsdim), dtype=th.float32
        )
        self.offset_by_ss = th.zeros(
            (n_subsets, self.max_rank), dtype=th.float32
        )
        for i, s in enumerate(self.subsets):
            for j, dim in enumerate(s.split(',')):
                self.psi_1_by_ss[i][j] = th.tensor(self.psi_1[int(dim)])
                self.psi_by_ss[i][j] = th.tensor(self.psi[int(dim)])
                self.offset_by_ss[i][j] = self.offset[int(dim)]

        device = cfg.device
        self.psi_1_by_ss = self.psi_1_by_ss.to(device)
        self.psi_by_ss = self.psi_by_ss.to(device)
        self.offset_by_ss = self.offset_by_ss.to(device)
        self.offset_1 = th.tensor(
            self.offset_1, device=device, dtype=th.float32
        )
        self.task = self.task.to(device)
        self.subgoal_idxs = []
        for s in self.subsets:
            self.subgoal_idxs.append([self.task_map[f] for f in s.split(',')])

    @staticmethod
    def parse_lo_info(cfg):
        p = Path(cfg.lo.init_from)
        abs_path = str(p.with_name(p.stem + '_abs.json'))
        log.debug(f'Loading subset information from checkpoint {abs_path}')
        with open(abs_path, 'rt') as ft:
            d = json.load(ft)
            if 'task_map' in d:
                task_map = d['task_map']
            cperf = d['cperf']
            if 'total' in cperf:
                del cperf['total']
        eps = cfg.goal_space.from_lo_eps
        subsets = sorted_nicely_sep(
            list(set([k for k, v in cperf.items() if v >= 1.0 - eps]))
        )
        if cfg.goal_space.rank_min > 0:
            subsets = [
                a
                for a in subsets
                if len(a.split(',')) >= cfg.goal_space.rank_min
            ]
        if cfg.goal_space.rank_max > 0:
            subsets = [
                a
                for a in subsets
                if len(a.split(',')) <= cfg.goal_space.rank_max
            ]
        return subsets, task_map

    def action_mask_hi(self):
        nt = self.action_space_hi['task'].n
        ng = self.action_space_hi['subgoal'].shape[0]
        mask = th.zeros(nt, ng)

        for i, feats in enumerate(self.subgoal_idxs):
            mask[i, feats] = 1
        for f in self.mask_gsfeats:
            mask[:, self.task_map[str(f)]] = 0

        return mask.to(self.task.device)

    def gs_obs(self, obs):
        return obs[self.gs_key]

    def translate(self, gs_obs, task, subgoal, delta_gs_obs=None):
        # Copy subgoal features to the front for compatibility with projections.
        subgoal_s = th.zeros(
            (subgoal.shape[0], self.max_rank), device=subgoal.device
        )
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        if self.delta_actions:
            # Subgoal is projected current state plus the specified action
            proj_obs = (
                th.bmm(
                    gs_obs.unsqueeze(1),
                    self.psi_by_ss.index_select(0, task).transpose(1, 2),
                ).squeeze(1)
                + self.offset_by_ss.index_select(0, task)
            )
            subgoal_s = proj_obs + subgoal_s

        # Backproject absolute subgoal into observation space
        bproj_goal = (
            th.bmm(
                subgoal_s.unsqueeze(1), self.psi_1_by_ss.index_select(0, task)
            ).squeeze(1)
            + self.offset_1
        )
        # Add delta features from current state; with delta actions, this has
        # already been taken care of via proj_obs.
        if delta_gs_obs is not None:
            raise RuntimeError('Umm no idea what this should do??')
            bproj_goal[:, self.delta_feats] += delta_gs_obs[:, self.delta_feats]
        elif not self.delta_actions:
            bproj_goal[:, self.delta_feats] += gs_obs[:, self.delta_feats]
        task_rep = self.task.index_select(0, task)
        # Give desired delta to ground truth
        goal = bproj_goal[:, self.task_idx] - gs_obs[:, self.task_idx]
        if len(self.twist_feats) > 0:
            twf = self.twist_feats
            goal[:, twf] = (
                th.remainder(
                    (
                        bproj_goal[:, self.task_idx][:, twf]
                        - gs_obs[:, self.task_idx][:, twf]
                    )
                    + np.pi,
                    2 * np.pi,
                )
                - np.pi
            )
        goal = goal * task_rep
        return {
            'desired_goal': goal,
            'task': task_rep,
        }

    # Update backprojected subgoal
    def update_bp_subgoal(self, gs_obs, next_gs_obs, action_hi):
        upd = (
            gs_obs[:, self.task_idx]
            - next_gs_obs[:, self.task_idx]
            + action_hi['desired_goal']
        ) * action_hi['task']
        if len(self.twist_feats) > 0:
            twf = self.twist_feats
            upd[:, twf] = (
                th.remainder(
                    (
                        gs_obs[:, self.task_idx][:, twf]
                        - next_gs_obs[:, self.task_idx][:, twf]
                        + action_hi['desired_goal'][:, twf]
                    )
                    + np.pi,
                    2 * np.pi,
                )
                - np.pi
            ) * action_hi['task'][:, twf]
        return upd

    def observation_lo(self, o_obs, action_hi):
        return {
            'observation': o_obs[:, self.obs_mask],
            'desired_goal': action_hi['desired_goal'],
            'task': action_hi['task'],
        }

    def observation_hi(self, obs):
        tobs = copy(obs)
        del tobs[self.gs_key]
        return tobs

    def dist_lo(self, gs_obs, task, subgoal):
        subgoal_s = th.zeros_like(subgoal)
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        proj_obs = (
            th.bmm(
                gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        return th.linalg.norm(subgoal_s - proj_obs, ord=2, dim=1)

    # Potential-based reward for low-level policy
    def reward_lo(self, gs_obs, next_gs_obs, task, subgoal):
        subgoal_s = th.zeros_like(subgoal)
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        proj_obs = (
            th.bmm(
                gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        proj_next_obs = (
            th.bmm(
                next_gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        d = th.linalg.norm(subgoal - proj_obs, ord=2, dim=1)
        dn = th.linalg.norm(subgoal - proj_next_obs, ord=2, dim=1)
        return d - dn

    def log_goal_spaces(self):
        log.info(f'Considering {len(self.subsets)} goal space subsets')
        for i, s in enumerate(self.subsets):
            name = ','.join(
                [
                    CtrlgsPreTrainingEnv.feature_name(
                        self.robot, self.features, int(f)
                    )
                    for f in s.split(',')
                ]
            )
            log.info(f'Subset {i}: {name} ({s})')


class HSD3Agent(Agent):
    '''
    A HRL agent that can leverage a low-level policy obtained via hierarchical
    skill discovery. It implements a composite action space for chosing (a) a
    goal space subset and (b) an actual goal. Soft Actor-Critic is used for
    performing policy and Q-function updates.

    The agent unfortunately depends on

    This implementation also features the following:
    - Dense high-level updates as in DynE
    '''

    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        if not hasattr(model, 'hi'):
            raise ValueError('Model needs "hi" module')
        if not hasattr(model, 'lo'):
            raise ValueError('Model needs "lo" module')
        if not hasattr(model.hi, 'pi_subgoal'):
            raise ValueError('Model needs "hi.pi_subgoal" module')
        if not hasattr(model.hi, 'pi_task'):
            raise ValueError('Model needs "hi.pi_task" module')
        if not hasattr(model.hi, 'q'):
            raise ValueError('Model needs "hi.q" module')
        if not hasattr(model.lo, 'pi'):
            raise ValueError('Model needs "lo.pi" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'HSD3Agent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'HSD3Agent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'HSD3Agent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(f'HSD3Agent requires a "observation" observation')
        if not cfg.goal_space.key in env.observation_space.spaces:
            raise ValueError(
                f'HSD3Agent requires a "{cfg.goal_space.key}" observation'
            )

        self._iface = HiToLoInterface(env, cfg)
        self._iface.log_goal_spaces()
        self._ckey = 'subgoal'
        self._action_space_c = self._iface.action_space_hi[self._ckey]
        self._dkey = 'task'
        self._action_space_d = self._iface.action_space_hi[self._dkey]

        self._model = model
        self._model_pi_c = model.hi.pi_subgoal
        self._model_pi_d = model.hi.pi_task
        self._optim_pi_c = optim.hi.pi_subgoal
        self._optim_pi_d = optim.hi.pi_task
        self._optim = optim
        self._bsz = int(cfg.batch_size)
        self._gamma = float(cfg.gamma)
        self._polyak = float(cfg.polyak)
        self._rpbuf_size = int(cfg.rpbuf_size)
        self._samples_per_update = int(cfg.samples_per_update)
        self._num_updates = int(cfg.num_updates)
        self._warmup_samples = int(cfg.warmup_samples)
        self._randexp_samples = int(cfg.randexp_samples)
        self._clip_grad_norm = float(cfg.clip_grad_norm)
        self._action_interval = int(cfg.action_interval)
        expectation_d = str(cfg.expectation_d)
        if expectation_d == 'full':
            self._expectation_d = -1
        elif expectation_d == 'False':
            self._expectation_d = 1
        else:
            self._expectation_d = int(expectation_d)

        self._dyne_updates = bool(cfg.dyne_updates)

        self._action_c_mask = self._iface.action_mask_hi().unsqueeze(0).float()
        self._target_entropy_factor_c = cfg.target_entropy_factor_c
        self._target_entropy_c = -1 * self._target_entropy_factor_c
        log_alpha_c = [
            np.log(cfg.alpha_c) for _ in range(self._action_c_mask.shape[1])
        ]
        if cfg.optim_alpha_c is None:
            self._log_alpha_c = th.tensor(
                log_alpha_c, device=cfg.device, dtype=th.float32
            )
        else:
            self._log_alpha_c = th.tensor(
                log_alpha_c,
                device=cfg.device,
                dtype=th.float32,
                requires_grad=True,
            )
            self._optim_alpha_c = hydra.utils.instantiate(
                cfg.optim_alpha_c, [self._log_alpha_c]
            )

        self._target_entropy_d = (
            np.log(self._action_space_d.n) * cfg.target_entropy_factor_d
        )
        self._uniform_entropy_d = np.log(self._action_space_d.n)
        log_alpha_d = np.log(cfg.alpha_d)
        if cfg.optim_alpha_d is None:
            self._log_alpha_d = th.tensor(log_alpha_d)
            self._optim_alpha_d = None
        else:
            self._log_alpha_d = th.tensor(log_alpha_d, requires_grad=True)
            self._optim_alpha_d = hydra.utils.instantiate(
                cfg.optim_alpha_d, [self._log_alpha_d]
            )

        log.info(
            f'Initializing low-level model from checkpoint {cfg.lo.init_from}'
        )
        with open(cfg.lo.init_from, 'rb') as fd:
            data = th.load(fd, map_location=th.device(cfg.device))
            if '_model' in data:  # Checkpoint
                missing_keys, _ = self._model.lo.load_state_dict(
                    data['_model'], strict=False
                )
            else:  # Raw model weights
                missing_keys, _ = self._model.lo.load_state_dict(
                    data, strict=False
                )
            if len(missing_keys) > 0:
                raise ValueError(f'Missing keys in model: {missing_keys}')

        rpbuf_device = cfg.rpbuf_device if cfg.rpbuf_device != 'auto' else None
        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs, device=rpbuf_device
        )
        self._staging = ReplayBuffer(
            size=max(2, self._action_interval) * env.num_envs,
            interleave=env.num_envs,
            device=rpbuf_device,
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []
        self._d_batchin = None
        self._onehots = None

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space_lo = env.action_space
        self._action_factor_lo = env.action_space.high[0]
        self._action_factor_c = self._action_space_c.high[0]
        self._obs_space = self._iface.observation_space_hi
        self._obs_keys = list(self._obs_space.spaces.keys())

        self._q_hi = self._model.hi.q
        self._pi_lo_det = DeterministicLo(self._model.lo.pi)
        if cfg.trace:
            self._q_hi = TracedModule(self._q_hi)
            self._pi_lo_det = TracedModule(self._pi_lo_det)

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha_c',
            '_optim_alpha_c',
            '_log_alpha_d',
            '_optim_alpha_d',
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        iface = HiToLoInterface(env, cfg)
        return {
            'lo': iface.observation_space_lo,
            'hi': {
                'pi_task': iface.observation_space_hi,
                'pi_subgoal': gym.spaces.Dict(
                    task=iface.action_space_hi.spaces['task'],
                    **iface.observation_space_hi.spaces,
                ),
                'q': iface.observation_space_hi,
            },
        }

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        iface = HiToLoInterface(env, cfg)
        return {
            'lo': env.action_space,
            'hi': {
                'pi_task': iface.action_space_hi['task'],
                'pi_subgoal': iface.action_space_hi['subgoal'],
                'q': iface.action_space_hi,
            },
        }

    def action_hi_d_qinput(self, action_d: th.Tensor) -> th.Tensor:
        nd = self._action_space_d.n
        return self._onehots.index_select(0, action_d.view(-1))

    def action_hi_rand(self, env, time):
        actions = [
            self._iface.action_space_hi.sample() for _ in range(env.num_envs)
        ]
        device = list(self._model.parameters())[0].device
        action_d = th.tensor(
            [a[self._dkey] for a in actions], dtype=th.int64, device=device
        )
        action_c = th.tensor(
            np.stack([a['subgoal'] for a in actions]), device=device
        )
        if self._action_c_mask is not None:
            mask = self._action_c_mask.index_select(1, action_d).squeeze(0)
            action_c = action_c * mask
        return {self._dkey: action_d, self._ckey: action_c}

    def action_hi_cd(self, env, obs):
        time = obs['time']
        obs_hi = self._iface.observation_hi(obs)
        obs_hi['time'] = th.zeros_like(obs_hi['time'])
        dist_d = self._model_pi_d(obs_hi)
        if self.training:
            dist_d = D.Categorical(logits=dist_d.logits)
            action_d = dist_d.sample()
        else:
            action_d = dist_d.logits.argmax(dim=1)

        subgoal_obs_hi = copy(obs_hi)
        nd = self._action_space_d.n
        subgoal_obs_hi[self._dkey] = (
            F.one_hot(action_d, nd).float().view(-1, nd)
        )
        dist_c = self._model_pi_c(subgoal_obs_hi)
        if self.training:
            action_c = dist_c.sample()
        else:
            action_c = dist_c.mean
        action_c = action_c * self._action_factor_c

        assert action_c.ndim == 3, 'Subgoal policy not multihead?'
        if self._action_c_mask is not None:
            action_c = action_c * self._action_c_mask
        action_c = dim_select(action_c, 1, action_d)

        return {self._dkey: action_d, self._ckey: action_c}

    def action_hi(self, env, obs, prev_action):
        if self._n_samples < self._randexp_samples and self.training:
            action = self.action_hi_rand(env, obs['time'])
        else:
            action = self.action_hi_cd(env, obs)
        return action

    def action_lo(self, env, obs):
        action = self._pi_lo_det(obs)
        action = action * self._action_factor_lo
        return action

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        step = obs['time'].remainder(self._action_interval).long().view(-1)
        keep_action_hi = step != 0

        def retain(x, y, mask):
            return mask * x + th.logical_not(mask) * y

        prev_gs_obs = env.ctx.get('gs_obs', None)
        action_hi = env.ctx.get('action_hi', None)
        obs_hi = copy(obs)

        tr_action_hi = env.ctx.get('tr_action_hi', None)
        if action_hi is None or not keep_action_hi.all().item():
            with th.no_grad():
                new_action_hi = self.action_hi(env, obs_hi, action_hi)
            tr_new_action_hi = self._iface.translate(
                self._iface.gs_obs(obs),
                new_action_hi[self._dkey],
                new_action_hi[self._ckey],
            )
            if action_hi is None:
                action_hi = deepcopy(new_action_hi)
                tr_action_hi = deepcopy(tr_new_action_hi)
            else:
                c = self._ckey
                d = self._dkey
                # Replace raw actions
                action_hi[d] = retain(
                    action_hi[d], new_action_hi[d], keep_action_hi
                )
                action_hi[c] = retain(
                    action_hi[c],
                    new_action_hi[c],
                    keep_action_hi.unsqueeze(1).expand_as(action_hi[c]),
                )
                # Replace translated actions
                tr_action_hi['task'] = retain(
                    tr_action_hi['task'],
                    tr_new_action_hi['task'],
                    keep_action_hi.unsqueeze(1).expand_as(tr_action_hi['task']),
                )
                tr_action_hi['desired_goal'] = self._iface.update_bp_subgoal(
                    prev_gs_obs, self._iface.gs_obs(obs), tr_action_hi
                )
                tr_action_hi['desired_goal'] = retain(
                    tr_action_hi['desired_goal'],
                    tr_new_action_hi['desired_goal'],
                    keep_action_hi.unsqueeze(1).expand_as(
                        tr_action_hi['desired_goal']
                    ),
                )
        else:
            tr_action_hi['desired_goal'] = self._iface.update_bp_subgoal(
                prev_gs_obs, self._iface.gs_obs(obs), tr_action_hi
            )

        env.ctx['action_hi'] = action_hi
        env.ctx['tr_action_hi'] = tr_action_hi
        if not 'gs_obs' in env.ctx:
            env.ctx['gs_obs'] = self._iface.gs_obs(obs).clone()
        else:
            env.ctx['gs_obs'].copy_(self._iface.gs_obs(obs))

        with th.no_grad():
            obs_lo = self._iface.observation_lo(
                obs['observation'], tr_action_hi
            )
            action_lo = self.action_lo(env, obs_lo)

        if self.training:
            return action_lo, {
                'action_hi': action_hi,
                'tr_action_hi': tr_action_hi,
                #'gs_obs0': env.ctx['gs_obs0'],
                'obs_hi': obs_hi,
            }

        # Additional visualization info for evals
        subsets = [
            self._iface.subsets[i.item()] for i in action_hi['task'].cpu()
        ]
        sg_cpu = action_hi['subgoal'].cpu().numpy()
        sgd_cpu = tr_action_hi['desired_goal'].cpu().numpy()
        subgoals = []
        subgoals_d = []
        for i in range(env.num_envs):
            n = len(subsets[i].split(','))
            subgoals.append(sg_cpu[i, :n])
            feats = [self._iface.task_map[f] for f in subsets[i].split(',')]
            subgoals_d.append(sgd_cpu[i, feats])
        return action_lo, {
            'action_hi': action_hi,
            'tr_action_hi': tr_action_hi,
            'obs_hi': obs_hi,
            'st': subsets,
            'sg': subgoals,
            'sgd': subgoals_d,
            'viz': ['st', 'sg', 'sgd'],
        }

    def step(
        self,
        env,
        obs,
        action,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, done, info = result
        action_hi = extra['action_hi']
        tr_action_hi = extra['tr_action_hi']
        obs_hi = extra['obs_hi']
        # Ignore terminal state if we have a timeout
        fell_over = th.zeros_like(done, device='cpu')
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                # log.info('Ignoring timeout')
                done[i] = False
            elif 'fell_over' in info[i]:
                fell_over[i] = True
        fell_over = fell_over.to(done.device)

        d = dict(
            terminal=done,
            step=obs['time'].remainder(self._action_interval).long(),
        )
        for k, v in action_hi.items():
            d[f'action_hi_{k}'] = v
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs_hi[k]
            if k != 'prev_task' and k != 'prev_subgoal':
                d[f'next_obs_{k}'] = next_obs[k]
        d['reward'] = reward

        self._staging.put_row(d)
        self._cur_rewards.append(reward)

        if self._staging.size == self._staging.max:
            self._staging_to_buffer()

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        ilv = self._staging.interleave
        if self._buffer.size + self._staging.size - ilv < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _staging_to_buffer(self):
        ilv = self._staging.interleave
        buf = self._staging
        assert buf._b is not None
        c = self._action_interval
        # Stack at least two transitions because for training the low-level
        # policy we'll need the next high-level action.
        n_stack = max(c, 2)
        batch: Dict[str, th.Tensor] = dict()
        idx = (
            buf.start + th.arange(0, ilv * n_stack, device=buf.device)
        ) % buf.max
        for k in set(buf._b.keys()):
            b = buf._b[k].index_select(0, idx)
            b = b.view((n_stack, ilv) + b.shape[1:]).transpose(0, 1)
            batch[k] = b

        # c = action_freq
        # i = batch['step']
        # Next action at c - i steps further, but we'll take next_obs so
        # access it at c - i - 1
        next_action_hi = (c - 1) - batch['step'][:, 0]
        # If we have a terminal before, use this instead
        terminal = batch['terminal'].clone()
        for j in range(1, c):
            terminal[:, j] |= terminal[:, j - 1]
        first_terminal = c - terminal.sum(dim=1)
        # Lastly, the episode could have ended with a timeout, which we can
        # detect if we took another action_hi (i == 0) prematurely. This will screw
        # up the reward summation, but hopefully it doesn't hurt too much.
        next_real_action_hi = th.zeros_like(next_action_hi) + c
        for j in range(1, c):
            idx = th.where(batch['step'][:, j] == 0)[0]
            next_real_action_hi[idx] = next_real_action_hi[idx].clamp(0, j - 1)
        next_idx = th.min(
            th.min(next_action_hi, first_terminal), next_real_action_hi
        )

        # Sum up discounted rewards until next c - i - 1
        reward = batch['reward'][:, 0].clone()
        for j in range(1, c):
            reward += self._gamma ** j * batch['reward'][:, j] * (next_idx >= j)

        not_done = th.logical_not(dim_select(batch['terminal'], 1, next_idx))
        obs = {k: batch[f'obs_{k}'][:, 0] for k in self._obs_keys}
        obs['time'] = batch['step'][:, 0:1].clone()
        obs_p = {
            k: dim_select(batch[f'next_obs_{k}'], 1, next_idx)
            for k in self._obs_keys
        }
        obs_p['time'] = obs_p['time'].clone().unsqueeze(1)
        obs_p['time'].fill_(0)

        gamma_exp = th.zeros_like(reward) + self._gamma
        gamma_exp.pow_(next_idx + 1)

        db = dict(
            reward=reward,
            not_done=not_done,
            terminal=batch['terminal'][:, 0],
            gamma_exp=gamma_exp,
        )
        db[f'action_hi_{self._dkey}'] = batch[f'action_hi_{self._dkey}'][:, 0]
        db[f'action_hi_{self._ckey}'] = batch[f'action_hi_{self._ckey}'][:, 0]
        for k, v in obs.items():
            db[f'obs_{k}'] = v
        for k, v in obs_p.items():
            db[f'next_obs_{k}'] = v

        self._buffer.put_row(db)

    def _update(self):
        def act_logp_c(obs, mask):
            dist = self._model_pi_c(obs)
            action = dist.rsample()
            if mask is not None:
                log_prob = (dist.log_prob(action) * mask).sum(
                    dim=-1
                ) / mask.sum(dim=-1)
                action = action * mask * self._action_factor_c
            else:
                log_prob = dist.log_prob(action).sum(dim=-1)
                action = action * self._action_factor_c
            return action, log_prob

        def q_target(batch):
            reward = batch['reward']
            not_done = batch['not_done']
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            alpha_c = self._log_alpha_c.detach().exp()
            alpha_d = self._log_alpha_d.detach().exp()
            bsz = reward.shape[0]
            d_batchin = self._d_batchin.narrow(0, 0, bsz * nd)
            c_batchmask = self._c_batchmask.narrow(0, 0, bsz * nd)

            dist_d = self._model_pi_d(obs_p)
            action_c, log_prob_c = act_logp_c(obs_p, self._action_c_mask)

            if self._expectation_d == -1 and nd > 1:
                # Present interleaved observation so that we can easily
                # reshape the result into BxA1xA2.
                obs_pe = {}
                for k, v in obs_p.items():
                    obs_pe[k] = v.repeat_interleave(nd, dim=0)
                obs_pe[self._dkey] = d_batchin
                obs_pe[self._ckey] = action_c.view(d_batchin.shape[0], -1)
                q_t = th.min(self._target.hi.q(obs_pe), dim=-1).values

                q_t = q_t.view(bsz, nd)
                log_prob_c = log_prob_c.view(bsz, nd)
                v_est = (dist_d.probs * (q_t - log_prob_c * alpha_c)).sum(
                    dim=-1
                ) + alpha_d * (dist_d.entropy() - self._uniform_entropy_d)
            else:
                action_d = th.multinomial(dist_d.probs, nds, replacement=True)
                log_prob_d = dist_d.logits.gather(1, action_d)

                obs_pe = {}
                for k, v in obs_p.items():
                    if nds > 1:
                        obs_pe[k] = v.repeat_interleave(nds, dim=0)
                    else:
                        obs_pe[k] = v
                obs_pe[self._dkey] = self.action_hi_d_qinput(action_d).view(
                    -1, nd
                )

                action_c = dim_select(action_c, 1, action_d).view(
                    -1, action_c.shape[-1]
                )
                log_prob_c = log_prob_c.gather(1, action_d)
                obs_pe[self._ckey] = action_c

                q_t = th.min(self._target.hi.q(obs_pe), dim=-1).values.view(
                    -1, nds
                )
                log_prob_c = log_prob_c.view(-1, nds)
                if self._action_c_mask is not None:
                    ac = alpha_c.index_select(0, action_d.view(-1)).view_as(
                        log_prob_c
                    )
                else:
                    ac = alpha_c
                v_est = (q_t - ac * log_prob_c - alpha_d * log_prob_d).mean(
                    dim=-1
                )

            return reward + batch['gamma_exp'] * not_done * v_est

        for p in self._model.parameters():
            mdevice = p.device
            break
        bsz = self._bsz
        nd = self._action_space_d.n
        nds = self._expectation_d
        if nd == 1:
            nds = 1
        if self._d_batchin is None:
            self._onehots = F.one_hot(th.arange(nd), nd).float().to(mdevice)
            self._d_batchin = self.action_hi_d_qinput(
                th.arange(bsz * nd).remainder(nd).to(mdevice)
            )
            if self._action_c_mask is not None:
                self._c_batchmask = self._action_c_mask.index_select(
                    1, th.arange(bsz * nd, device=mdevice).remainder(nd)
                ).squeeze(0)
            else:
                self._c_batchmask = None

        if not self._dyne_updates:
            assert (
                self._buffer.start == 0 or self._buffer.size == self._buffer.max
            )
            indices = th.where(
                self._buffer._b['obs_time'][: self._buffer.size] == 0
            )[0]
        gbatch = None
        if self._dyne_updates and self._bsz < 512:
            gbatch = self._buffer.get_batch(
                self._bsz * self._num_updates,
                device=mdevice,
            )

        for i in range(self._num_updates):
            if self._dyne_updates:
                if gbatch is not None:
                    batch = {
                        k: v.narrow(0, i * self._bsz, self._bsz)
                        for k, v in gbatch.items()
                    }
                else:
                    batch = self._buffer.get_batch(
                        self._bsz,
                        device=mdevice,
                    )
            else:
                batch = self._buffer.get_batch_where(
                    self._bsz, indices=indices, device=mdevice
                )

            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            alpha_c = self._log_alpha_c.detach().exp()
            alpha_d = self._log_alpha_d.detach().exp()

            # Backup for Q-Function
            with th.no_grad():
                backup = q_target(batch)

            # Q-Function update
            q_in = copy(obs)
            q_in[self._dkey] = self.action_hi_d_qinput(
                batch[f'action_hi_{self._dkey}']
            )
            q_in[self._ckey] = batch[f'action_hi_{self._ckey}']
            q = self._q_hi(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup, reduction='none')
            q2_loss = F.mse_loss(q2, backup, reduction='none')
            q_loss = q1_loss.mean() + q2_loss.mean()
            self._optim.hi.q.zero_grad()
            q_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.q.parameters(), self._clip_grad_norm
                )
            self._optim.hi.q.step()

            # Policy update
            for param in self._model.hi.q.parameters():
                param.requires_grad_(False)

            # No time input for policy, and Q-functions are queried as if step
            # would be 0 (i.e. we would take an action)
            obs['time'] = obs['time'] * 0
            dist_d = self._model_pi_d(obs)
            action_c, log_prob_c = act_logp_c(obs, self._action_c_mask)

            if self._expectation_d == -1 and nd > 1:
                obs_e = {}
                for k, v in obs.items():
                    obs_e[k] = v.repeat_interleave(nd, dim=0)
                obs_e[self._dkey] = self._d_batchin
                obs_e[self._ckey] = action_c.view(self._d_batchin.shape[0], -1)
                q = th.min(self._q_hi(obs_e), dim=-1).values

                q = q.view(bsz, nd)
                log_prob_c = log_prob_c.view(bsz, nd)
                pi_loss = (dist_d.probs * (alpha_c * log_prob_c - q)).sum(
                    dim=-1
                ) - alpha_d * (dist_d.entropy() - self._uniform_entropy_d)
            else:
                action_d = th.multinomial(dist_d.probs, nds, replacement=True)
                log_prob_d = dist_d.logits.gather(1, action_d)

                obs_e = {}
                for k, v in obs.items():
                    if nds > 1:
                        obs_e[k] = v.repeat_interleave(nds, dim=0)
                    else:
                        obs_e[k] = v
                obs_e[self._dkey] = self.action_hi_d_qinput(action_d).view(
                    -1, nd
                )

                action_c = dim_select(action_c, 1, action_d).view(
                    -1, action_c.shape[-1]
                )
                log_prob_co = log_prob_c
                log_prob_c = log_prob_c.gather(1, action_d)
                obs_e[self._ckey] = action_c

                q = th.min(self._q_hi(obs_e), dim=-1).values.view(-1, nds)
                log_prob_c = log_prob_c.view(-1, nds)
                if self._action_c_mask is not None:
                    ac = alpha_c.index_select(0, action_d.view(-1)).view_as(
                        log_prob_c
                    )
                else:
                    ac = alpha_c
                pi_loss = (ac * log_prob_c + alpha_d * log_prob_d - q).mean(
                    dim=-1
                )

            pi_loss = pi_loss.mean()
            self._optim_pi_c.zero_grad()
            self._optim_pi_d.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model_pi_c.parameters(), self._clip_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self._model_pi_d.parameters(), self._clip_grad_norm
                )
            self._optim_pi_c.step()
            self._optim_pi_d.step()

            for param in self._model.hi.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha_c:
                if self._expectation_d != -1:
                    alpha_loss_c = (
                        -(
                            self._log_alpha_c.exp()
                            * dist_d.probs.detach()
                            * (
                                log_prob_co.detach() + self._target_entropy_c
                            ).view(bsz, nd)
                        )
                        .sum(dim=-1)
                        .mean()
                    )
                else:
                    alpha_loss_c = (
                        -(
                            self._log_alpha_c.exp()
                            * dist_d.probs.detach()
                            * (
                                log_prob_c.detach() + self._target_entropy_c
                            ).view(bsz, nd)
                        )
                        .sum(dim=-1)
                        .mean()
                    )
                self._optim_alpha_c.zero_grad()
                alpha_loss_c.backward()
                self._optim_alpha_c.step()
            if self._optim_alpha_d:
                alpha_loss_d = (
                    self._log_alpha_d.exp()
                    * (
                        dist_d.entropy().mean().cpu() - self._target_entropy_d
                    ).detach()
                )
                self._optim_alpha_d.zero_grad()
                alpha_loss_d.backward()
                self._optim_alpha_d.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.hi.q.parameters(),
                    self._model.hi.q.parameters(),
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('LossHi/Policy', pi_loss.item())
        self.tbw_add_scalar('LossHi/QValue', q_loss.item())
        with th.no_grad():
            bvar = backup.var()
            resvar1 = (backup - q1).var() / bvar
            resvar2 = (backup - q2).var() / bvar
        self.tbw_add_scalar('HealthHi/ResidualVariance1', resvar1.item())
        self.tbw_add_scalar('HealthHi/ResidualVariance2', resvar2.item())
        self.tbw_add_scalar('HealthHi/EntropyC', -log_prob_c.mean())
        self.tbw_add_scalar('HealthHi/EntropyD', dist_d.entropy().mean())
        if self._optim_alpha_c:
            self.tbw_add_scalar(
                'HealthHi/AlphaC', self._log_alpha_c.exp().mean().item()
            )
        if self._optim_alpha_d:
            self.tbw_add_scalar(
                'HealthHi/AlphaD', self._log_alpha_d.exp().item()
            )
        if self._n_updates % 10 == 1:
            self.tbw.add_histogram(
                'HealthHi/PiD',
                th.multinomial(
                    dist_d.probs,
                    int(np.ceil(1000 / self._bsz)),
                    replacement=True,
                ).view(-1),
                self._n_samples,
                bins=nd,
            )
        if self._n_updates % 100 == 1:
            self.tbw.add_scalars(
                'HealthHi/GradNorms',
                {
                    k: v.grad.norm().item()
                    for k, v in self._model.named_parameters()
                    if v.grad is not None
                },
                self.n_samples,
            )

        td_err1 = q1_loss.sqrt().mean().item()
        td_err2 = q2_loss.sqrt().mean().item()
        td_err = (td_err1 + td_err2) / 2
        self.tbw_add_scalar('HealthHi/AbsTDErrorTrain', td_err)
        self.tbw_add_scalar('HealthHi/AbsTDErrorTrain1', td_err1)
        self.tbw_add_scalar('HealthHi/AbsTDErrorTrain2', td_err2)

        avg_cr = th.cat(self._cur_rewards).mean().item()
        log_stats = [
            ('Sample', f'{self._n_samples}'),
            ('hi: up', f'{self._n_updates*self._num_updates}'),
            ('avg rew', f'{avg_cr:+0.3f}'),
            ('pi loss', f'{pi_loss.item():+.03f}'),
            ('q loss', f'{q_loss.item():+.03f}'),
            (
                'entropy',
                f'{-log_prob_c.mean().item():.03f},{dist_d.entropy().mean().item():.03f}',
            ),
            (
                'alpha',
                f'{self._log_alpha_c.mean().exp().item():.03f},{self._log_alpha_d.exp().item():.03f}',
            ),
        ]
        log.info(', '.join((f'{k} {v}' for k, v in log_stats)))
