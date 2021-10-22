# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.utils import dim_select

log = logging.getLogger(__name__)


def _parse_list(s, dtype):
    if s is None:
        return []
    return list(map(dtype, str(s).split('#')))


class SACHRLAgent(Agent):
    '''
    A generic HRL SAC agent, in which the high-level action space is continuous.
    It includes the optimization when dealing with temporal abstraction proposed
    in "Dynamics-Aware Embeddings".
    This agent doesn't provide a means to actually train the low-level agent; it
    is assumed to be fixed and will be loaded from a checkpoint instead.
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
        if not hasattr(model.hi, 'pi'):
            raise ValueError('Model needs "hi.pi" module')
        if not hasattr(model.hi, 'q'):
            raise ValueError('Model needs "hi.q" module')
        if not hasattr(model.lo, 'pi'):
            raise ValueError('Model needs "lo.pi" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'SACHRLAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'SACHRLAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'SACHRLAgent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(
                f'SACHRLAgent requires a "observation" observation'
            )

        self._model = model
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
        n_actions_hi = int(cfg.n_actions_hi)

        hide_from_lo = _parse_list(cfg.hide_from_lo, int)
        obs_space = env.observation_space.spaces['observation']
        assert len(obs_space.shape) == 1
        self._features_lo = list(range(obs_space.shape[0]))
        for f in hide_from_lo:
            self._features_lo.remove(f)

        self._target_entropy_factor = cfg.target_entropy_factor
        self._target_entropy = (
            -np.prod(n_actions_hi) * cfg.target_entropy_factor
        )
        log_alpha = np.log(cfg.alpha)
        if cfg.optim_alpha is None:
            self._log_alpha = th.tensor(log_alpha)
            self._optim_alpha = None
        else:
            self._optim_alpha = cfg.optim_alpha
            self._log_alpha = th.tensor(log_alpha, requires_grad=True)
            self._optim_alpha = hydra.utils.instantiate(
                cfg.optim_alpha, [self._log_alpha]
            )

        rpbuf_device = cfg.rpbuf_device if cfg.rpbuf_device != 'auto' else None
        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs, device=rpbuf_device
        )
        self._staging = ReplayBuffer(
            size=self._action_interval * env.num_envs,
            interleave=env.num_envs,
            device=rpbuf_device,
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []
        self._d_batchin = None

        log.info(f'Loading low-level model from {cfg.init_lo_from}')
        for p in self._model.parameters():
            mdevice = p.device
            break
        with open(cfg.init_lo_from, 'rb') as fd:
            data = th.load(fd, map_location=th.device(mdevice))
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

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space_lo = env.action_space
        self._action_factor_lo = env.action_space.high[0]
        as_hi = self.effective_action_space(env, cfg)['hi']
        self._action_space_hi = as_hi
        self._obs_space = env.observation_space
        self._obs_keys = list(self._obs_space.spaces.keys())

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha',
            '_optim_alpha',
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        hide_from_lo = _parse_list(cfg.hide_from_lo, int)
        obs_space = env.observation_space.spaces['observation']
        assert len(obs_space.shape) == 1
        features_lo = list(range(obs_space.shape[0]))
        for f in hide_from_lo:
            features_lo.remove(f)

        spaces = {}
        spaces['hi'] = env.observation_space
        lo_min = env.observation_space['observation'].low[features_lo]
        lo_max = env.observation_space['observation'].low[features_lo]
        # XXX The order of the observation spaces is important since e.g. with
        # DIAYN, we train the model on cat([observation,condition])
        spaces['lo'] = gym.spaces.Dict(
            [
                ('observation', gym.spaces.Box(lo_min, lo_max)),
                (
                    'hi_action',
                    SACHRLAgent.effective_action_space(env, cfg)['hi'],
                ),
            ]
        )
        return spaces

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        n_actions_hi = int(cfg.n_actions_hi)
        spaces = {}
        spaces['lo'] = env.action_space
        spaces['hi'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_actions_hi,), dtype=np.float32
        )
        return spaces

    def action_hi(self, env, obs) -> Tuple[th.Tensor, th.Tensor]:
        step = obs['time'].remainder(self._action_interval).long().view(-1)
        keep_action = step != 0

        action = env.ctx.get('action_hi', None)
        if action is None and keep_action.any().item():
            raise RuntimeError('Need to take first action at time=0')
        if action is None or not keep_action.all().item():
            if self._n_samples < self._randexp_samples and self.training:
                new_action = th.stack(
                    [
                        th.from_numpy(self._action_space_hi.sample())
                        for i in range(env.num_envs)
                    ]
                ).to(list(self._model.parameters())[0].device)
            else:
                obs_wo_time = copy(obs)
                obs_wo_time['time'] = th.zeros_like(obs['time'])
                dist = self._model.hi.pi(obs_wo_time)
                assert (
                    dist.has_rsample
                ), f'rsample() required for hi-level policy distribution'
                if self.training:
                    new_action = dist.sample()
                else:
                    new_action = dist.mean
            if action is None:
                action = new_action
            else:
                m = keep_action.unsqueeze(1)
                action = m * action + th.logical_not(m) * new_action
        env.ctx['action_hi'] = action

        return action, th.logical_not(keep_action)

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        with th.no_grad():
            action_hi, took_action_hi = self.action_hi(env, obs)

            obs_lo = {
                'observation': obs['observation'][:, self._features_lo],
                'hi_action': action_hi,
            }
            dist = self._model.lo.pi(obs_lo)
            action_lo = dist.mean * self._action_factor_lo

        return action_lo, {'a_hi': action_hi, 'viz': ['a_hi']}

    def step(
        self,
        env,
        obs,
        action,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, done, info = result
        # Ignore terminal state if we have a timeout
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                done[i] = False

        d = dict(
            reward=reward,
            terminal=done,
            step=obs['time'].remainder(self._action_interval).long(),
            action=extra['a_hi'],
        )
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs[k]
            d[f'next_obs_{k}'] = next_obs[k]

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
        batch: Dict[str, th.Tensor] = dict()
        idx = buf.start + th.arange(0, ilv, device=buf.device)
        for k in set(buf._b.keys()):
            s = [
                buf._b[k].index_select(0, (idx + i * ilv) % buf.max)
                for i in range(c)
            ]
            batch[k] = th.stack(s, dim=1)

        # c = action_freq
        # i = batch['step']
        # Next action at c - i steps further, but we'll take next_obs so
        # access it at c - i - 1
        next_action = (c - 1) - batch['step'][:, 0]
        # If we have a terminal before, use this instead
        terminal = batch['terminal'].clone()
        for j in range(1, c):
            terminal[:, j] |= terminal[:, j - 1]
        first_terminal = c - terminal.sum(dim=1)
        # Lastly, the episode could have ended with a timeout, which we can
        # detect if we took another action (i == 0) prematurely. This will screw
        # up the reward summation, but hopefully it doesn't hurt too much.
        next_real_action = th.zeros_like(next_action) + c
        for j in range(1, c):
            idx = th.where(batch['step'][:, j] == 0)[0]
            next_real_action[idx] = next_real_action[idx].clamp(0, j - 1)
        next_idx = th.min(th.min(next_action, first_terminal), next_real_action)

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
            gamma_exp=gamma_exp,
            action=batch['action'][:, 0],
        )
        for k, v in obs.items():
            db[f'obs_{k}'] = v
        for k, v in obs_p.items():
            db[f'next_obs_{k}'] = v

        self._buffer.put_row(db)

    def _update(self):
        for p in self._model.parameters():
            mdevice = p.device
            break

        def act_logp(obs):
            dist = self._model.hi.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

        for _ in range(self._num_updates):
            batch = self._buffer.get_batch(
                self._bsz,
                device=mdevice,
            )

            reward = batch['reward']
            not_done = batch['not_done']
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._target.hi.q(q_in), dim=-1).values
                backup = reward + batch['gamma_exp'] * not_done * (
                    q_tgt - self._log_alpha.detach().exp() * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch['action'], **obs)
            q = self._model.hi.q(q_in)
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
            a, log_prob = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(self._model.hi.q(q_in), dim=-1).values
            pi_loss = (self._log_alpha.detach().exp() * log_prob - q).mean()
            self._optim.hi.pi.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.pi.parameters(), self._clip_grad_norm
                )
            self._optim.hi.pi.step()

            for param in self._model.hi.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha:
                alpha_loss = -(
                    self._log_alpha.exp()
                    * (log_prob.mean().cpu() + self._target_entropy).detach()
                )
                self._optim_alpha.zero_grad()
                alpha_loss.backward()
                self._optim_alpha.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.hi.q.parameters(),
                    self._model.hi.q.parameters(),
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        self.tbw_add_scalar('Health/Entropy', -log_prob.mean())
        if self._optim_alpha:
            self.tbw_add_scalar('Health/Alpha', self._log_alpha.exp().item())
        if self._n_updates % 100 == 1:
            self.tbw.add_scalars(
                'Health/GradNorms',
                {
                    k: v.grad.norm().item()
                    for k, v in self._model.named_parameters()
                    if v.grad is not None
                },
                self.n_samples,
            )

        avg_cr = th.cat(self._cur_rewards).mean().item()
        log.info(
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {self._log_alpha.exp().item():.03f}'
        )
