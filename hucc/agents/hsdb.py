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
from hucc.agents.hsd3 import DeterministicLo, HiToLoInterface
from hucc.models import TracedModule
from hucc.utils import dim_select

log = logging.getLogger(__name__)


def _parse_list(s, dtype):
    if s is None:
        return []
    return list(map(dtype, str(s).split('#')))


class HSDBAgent(Agent):
    '''
    A HRL agent that can leverage a low-level policy obtained via hierarchical
    skill discovery. A bandit is used to select and learn the best goal space.

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
        if not hasattr(model.hi, 'q'):
            raise ValueError('Model needs "hi.q" module')
        if not hasattr(model.lo, 'pi'):
            raise ValueError('Model needs "lo.pi" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'HSDBAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'HSDBAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'HSDBAgent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(f'HSDBAgent requires a "observation" observation')
        if not cfg.goal_space.key in env.observation_space.spaces:
            raise ValueError(
                f'HSDBAgent requires a "{cfg.goal_space.key}" observation'
            )

        self._iface = HiToLoInterface(env, cfg)
        self._iface.log_goal_spaces()
        self._ckey = 'subgoal'
        self._action_space_c = self._iface.action_space_hi[self._ckey]
        self._dkey = 'task'
        self._action_space_d = self._iface.action_space_hi[self._dkey]

        self._model = model
        self._model_pi_c = model.hi.pi_subgoal
        self._bandit_d = hydra.utils.instantiate(
            cfg.bandit_d, self._action_space_d
        )
        # Start with pulling each bandit arm once
        self._initial_actions_d = list(range(self._action_space_d.n))[::-1]
        self._optim_pi_c = optim.hi.pi_subgoal
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

        log.info(
            f'Initializing low-level model from checkpoint {cfg.lo.init_from}'
        )
        with open(cfg.lo.init_from, 'rb') as fd:
            data = th.load(fd, map_location=th.device(cfg.device))
            missing_keys, _ = self._model.lo.load_state_dict(
                data['_model'], strict=False
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
            '_bandit_d',
            '_initial_actions_d',
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        iface = HiToLoInterface(env, cfg)
        return {
            'lo': iface.observation_space_lo,
            'hi': {
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
                'pi_subgoal': iface.action_space_hi['subgoal'],
                'q': iface.action_space_hi,
            },
        }

    def action_hi_d_qinput(self, action_d: th.Tensor) -> th.Tensor:
        nd = self._action_space_d.n
        return self._onehots.index_select(0, action_d.view(-1))

    def action_hi_rand(self, env, time):
        time_c = time.view(-1).cpu()
        actions = [
            self._iface.action_space_hi.sample() for _ in range(env.num_envs)
        ]
        if len(self._initial_actions_d) > 0:
            for i in range(env.num_envs):
                if time_c[i] == 0:
                    try:
                        actions[i][self._dkey] = self._initial_actions_d.pop()
                    except IndexError:
                        break

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
        time_c = obs['time'].view(-1).cpu()
        obs_hi = self._iface.observation_hi(obs)
        obs_hi['time'] = th.zeros_like(obs_hi['time'])

        device = list(self._model.parameters())[0].device
        if self.training:
            actions_dp = []
            for i in range(env.num_envs):
                if time_c[i] == 0 and len(self._initial_actions_d) > 0:
                    actions_dp.append(self._initial_actions_d.pop())
                else:
                    actions_dp.append(self._action_space_d.sample())
            action_d = th.tensor(actions_dp, dtype=th.long, device=device)
        else:
            action_d = th.stack(
                [self._bandit_d.best_action() for _ in range(env.num_envs)]
            ).to(device)

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

        # Update bandit
        for i, d in enumerate(done.cpu().numpy()):
            if d:
                self._bandit_d.update(
                    action_hi[self._dkey][i].cpu(), info[i]['reward_acc']
                )

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
            bsz = reward.shape[0]

            action_c, log_prob_c = act_logp_c(obs_p, self._action_c_mask)

            action_d = batch[f'action_hi_{self._dkey}']
            obs_p[self._dkey] = self.action_hi_d_qinput(action_d).view(-1, nd)

            action_c = dim_select(action_c, 1, action_d).view(
                -1, action_c.shape[-1]
            )
            log_prob_c = dim_select(log_prob_c, 1, action_d)
            obs_p[self._ckey] = action_c

            q_t = th.min(self._target.hi.q(obs_p), dim=-1).values
            if self._action_c_mask is not None:
                ac = alpha_c.index_select(0, action_d)
            else:
                ac = alpha_c
            v_est = q_t - ac * log_prob_c

            return reward + batch['gamma_exp'] * not_done * v_est

        for p in self._model.parameters():
            mdevice = p.device
            break
        bsz = self._bsz
        nd = self._action_space_d.n
        if self._onehots is None:
            self._onehots = F.one_hot(th.arange(nd), nd).float().to(mdevice)

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
            action_c, log_prob_c = act_logp_c(obs, self._action_c_mask)
            action_d = batch[f'action_hi_{self._dkey}']
            obs[self._dkey] = self.action_hi_d_qinput(action_d).view(-1, nd)

            action_c = dim_select(action_c, 1, action_d).view(
                -1, action_c.shape[-1]
            )
            log_prob_c = dim_select(log_prob_c, 1, action_d)
            obs[self._ckey] = action_c

            q = th.min(self._q_hi(obs), dim=-1).values
            if self._action_c_mask is not None:
                ac = alpha_c.index_select(0, action_d)
            else:
                ac = alpha_c
            pi_loss = ac * log_prob_c - q

            pi_loss = pi_loss.mean()
            self._optim_pi_c.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model_pi_c.parameters(), self._clip_grad_norm
                )
            self._optim_pi_c.step()

            for param in self._model.hi.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha_c:
                log_alpha = self._log_alpha_c.index_select(0, action_d)
                alpha_loss_c = -(
                    log_alpha.exp()
                    * (log_prob_c.view(-1).detach() + self._target_entropy_c)
                ).mean()
                self._optim_alpha_c.zero_grad()
                alpha_loss_c.backward()
                self._optim_alpha_c.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.hi.q.parameters(),
                    self._model.hi.q.parameters(),
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        dist_d = self._bandit_d.dist()

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
        self.tbw_add_scalar('HealthHi/EntropyD', dist_d.entropy())
        if self._optim_alpha_c:
            self.tbw_add_scalar(
                'HealthHi/AlphaC', self._log_alpha_c.exp().mean().item()
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
                f'{-log_prob_c.mean().item():.03f},{dist_d.entropy().item():.03f}',
            ),
            ('alpha', f'{self._log_alpha_c.mean().exp().item():.03f}'),
        ]
        log.info(', '.join((f'{k} {v}' for k, v in log_stats)))
