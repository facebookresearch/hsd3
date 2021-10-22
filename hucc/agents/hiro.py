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
from torch.distributions import Normal
from torch.nn import functional as F

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.utils import dim_select

log = logging.getLogger(__name__)


def _parse_list(s, dtype):
    if s is None:
        return []
    return list(map(dtype, str(s).split('#')))


class HIROAgent(Agent):
    '''
    A HIRO-like agent, but using SAC instead of TD3. It uses goal relabeling as
    proposed in the original HIRO paper.

    Environment rewards for the high-level policy are simply summed up and
    divided by action_interval_hi.
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
        if not hasattr(model.lo, 'q'):
            raise ValueError('Model needs "lo.q" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'HIROAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'HIROAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'HIROAgent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(f'HIROAgent requires a "observation" observation')

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
        self._dense_hi_updates = bool(cfg.dense_hi_updates)

        self._relabel_goals = bool(cfg.relabel_goals)
        self._action_interval_hi = int(cfg.action_interval_hi)
        self._goal_features = _parse_list(cfg.goal_space.features, int)
        self._gspace_min = _parse_list(cfg.goal_space.range_min, float)
        self._gspace_max = _parse_list(cfg.goal_space.range_max, float)
        subset = _parse_list(cfg.goal_space.feature_subset, int)
        if subset:
            self._goal_features = [self._goal_features[i] for i in subset]
            self._gspace_min = [self._gspace_min[i] for i in subset]
            self._gspace_max = [self._gspace_max[i] for i in subset]

        self._gspace_min_th = None
        self._gspace_max_th = None
        assert len(self._goal_features) == len(self._gspace_min)
        assert len(self._goal_features) == len(self._gspace_max)
        self._gspace_key = cfg.goal_space.key

        self._target_entropy_lo = (
            -np.prod(env.action_space.shape) * cfg.target_entropy_factor_lo
        )
        # Optimize log(alpha) so that we'll always have a positive factor
        log_alpha_lo = np.log(cfg.alpha_lo)
        if cfg.optim_alpha_lo is None:
            self._log_alpha = th.tensor(log_alpha_lo)
            self._optim_alpha_lo = None
        else:
            self._log_alpha_lo = th.tensor(log_alpha_lo, requires_grad=True)
            self._optim_alpha_lo = hydra.utils.instantiate(
                cfg.optim_alpha_lo, [self._log_alpha_lo]
            )

        self._target_entropy_hi = (
            -len(self._goal_features) * cfg.target_entropy_factor_hi
        )
        log_alpha_hi = np.log(cfg.alpha_hi)
        if cfg.optim_alpha_hi is None:
            self._log_alpha = th.tensor(log_alpha_hi)
            self._optim_alpha_hi = None
        else:
            self._log_alpha_hi = th.tensor(log_alpha_hi, requires_grad=True)
            self._optim_alpha_hi = hydra.utils.instantiate(
                cfg.optim_alpha_hi, [self._log_alpha_hi]
            )

        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []
        self._cur_rewards_lo: List[th.Tensor] = []

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space_lo = env.action_space
        self._action_factor_lo = env.action_space.high[0]
        as_hi = self.effective_action_space(env, cfg)['hi']
        self._action_space_hi = as_hi
        self._action_scale_hi = (as_hi.high - as_hi.low) / 2
        self._action_scale_hi_th = None
        self._action_center_hi = as_hi.low + self._action_scale_hi
        self._action_center_hi_th = None
        self._obs_space = env.observation_space
        self._obs_keys = list(self._obs_space.spaces.keys())
        self._obs_keys.remove('time')
        if cfg.goal_space.key != 'observation':
            self._obs_keys.remove(cfg.goal_space.key)
        self._normalize_reward_lo = bool(cfg.normalize_reward_lo)
        self._reward_lo = cfg.reward_lo
        self._ctrl_cost_lo = float(cfg.ctrl_cost_lo)
        self._fallover_penalty_lo = float(cfg.fallover_penalty_lo)

        # Masks for low-level features
        self._obs_lo_keys = list(self._obs_space.spaces.keys())
        if cfg.lo_obs is not None:
            self._obs_lo_keys = [cfg.lo_obs]
        else:
            self._obs_lo_keys.remove('time')
            if cfg.goal_space.key != 'observation':
                self._obs_lo_keys.remove(cfg.goal_space.key)
        hide_from_lo = _parse_list(cfg.hide_from_lo, str)
        self._obs_lo_mask: Dict[str, th.Tensor] = {}
        for hide in hide_from_lo:
            if ':' in hide:
                key = hide.split(':')[0]
                mask = th.ones(self._obs_space.spaces[key].shape)
                for f in map(int, hide.split(':')[1].split('-')):
                    mask[f] = 0
                self._obs_lo_mask[key] = mask
            else:
                self._obs_lo_keys.remove(hide)

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha_lo',
            '_optim_alpha_lo',
            '_log_alpha_hi',
            '_optim_alpha_hi',
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        d = copy(env.observation_space.spaces)
        del d['time']
        if cfg.goal_space.key != 'observation':
            del d[cfg.goal_space.key]
        hide_from_lo = _parse_list(cfg.hide_from_lo, str)
        for hide in hide_from_lo:
            if ':' in hide:
                # For simplicity we'll just multiply the hidden features with
                # zero.
                pass
            else:
                del d[hide]
        d['desired_goal'] = HIROAgent.effective_action_space(env, cfg)['hi']
        spaces: Dict[str, gym.spaces.Space] = {}
        spaces['lo'] = gym.spaces.Dict(d)

        d = copy(env.observation_space.spaces)
        if not cfg.dense_hi_updates:
            del d['time']
        if cfg.goal_space.key != 'observation':
            del d[cfg.goal_space.key]
        spaces['hi'] = gym.spaces.Dict(d)

        return spaces

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        gspace_min = _parse_list(cfg.goal_space.range_min, float)
        gspace_max = _parse_list(cfg.goal_space.range_max, float)
        subset = _parse_list(cfg.goal_space.feature_subset, int)
        if subset:
            gspace_min = [gspace_min[i] for i in subset]
            gspace_max = [gspace_max[i] for i in subset]

        spaces = {}
        spaces['lo'] = env.action_space
        spaces['hi'] = gym.spaces.Box(
            low=np.array(gspace_min, dtype=np.float32),
            high=np.array(gspace_max, dtype=np.float32),
        )
        return spaces

    def scale_action_hi(self, action: th.Tensor) -> th.Tensor:
        if self._action_scale_hi_th is None:
            self._action_scale_hi_th = th.tensor(self._action_scale_hi).to(
                action.device
            )
            self._action_center_hi_th = th.tensor(self._action_center_hi).to(
                action.device
            )
        return (action * self._action_scale_hi_th) + self._action_center_hi_th

    def action_hi(self, env, obs) -> Tuple[th.Tensor, th.Tensor]:
        step = obs['time'].remainder(self._action_interval_hi).long().view(-1)
        keep_action = step != 0
        gstate_new = obs[self._gspace_key][:, self._goal_features]

        action = env.ctx.get('action_hi', None)
        gstate = env.ctx.get('gstate_hi', None)
        if action is None and keep_action.any().item():
            raise RuntimeError('Need to take first action at time=0')
        # Goal transition
        if gstate is not None:
            action = gstate + action - gstate_new
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
                if self._dense_hi_updates:
                    obs_wo_time['time'] = th.zeros_like(obs_wo_time['time'])
                else:
                    del obs_wo_time['time']
                dist = self._model.hi.pi(obs_wo_time)
                assert (
                    dist.has_rsample
                ), f'rsample() required for hi-level policy distribution'
                if self.training:
                    new_action = dist.sample()
                else:
                    new_action = dist.mean
                new_action = self.scale_action_hi(new_action)
            if action is None:
                action = new_action
            else:
                m = keep_action.unsqueeze(1)
                action = m * action + th.logical_not(m) * new_action
        env.ctx['action_hi'] = action
        env.ctx['gstate_hi'] = gstate_new

        return action, th.logical_not(keep_action)

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        with th.no_grad():
            action_hi, took_action_hi = self.action_hi(env, obs)

            if self._n_samples < self._randexp_samples and self.training:
                action_lo = th.stack(
                    [
                        th.from_numpy(self._action_space_lo.sample())
                        for i in range(env.num_envs)
                    ]
                ).to(list(self._model.parameters())[0].device)
            else:
                obs_lo = {k: obs[k] for k in self._obs_lo_keys}
                for k, m in self._obs_lo_mask.items():
                    m = m.to(obs_lo[k])
                    obs_lo[k] = obs_lo[k] * m
                    self._obs_lo_mask[k] = m
                obs_lo['desired_goal'] = action_hi
                dist = self._model.lo.pi(obs_lo)
                assert (
                    dist.has_rsample
                ), f'rsample() required for lo-level policy distribution'
                if self.training:
                    action_lo = dist.sample() * self._action_factor_lo
                else:
                    action_lo = dist.mean * self._action_factor_lo

        return action_lo, {'a_hi': action_hi, 'viz': ['a_hi']}

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, done, info = result
        # Ignore terminal state if we have a timeout
        fell_over = th.zeros_like(done, device='cpu')
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                done[i] = False
            elif 'fell_over' in info[i]:
                fell_over[i] = True
        fell_over = fell_over.to(done.device)

        action_lo = action
        action_hi = extra['a_hi']
        gs_obs = obs[self._gspace_key][:, self._goal_features]
        gs_next_obs = next_obs[self._gspace_key][:, self._goal_features]
        auto_next_action_hi = gs_obs + action_hi - gs_next_obs
        if self._normalize_reward_lo:
            if self._action_scale_hi_th is None:
                self._action_scale_hi_th = th.tensor(self._action_scale_hi).to(
                    action.device
                )
                self._action_center_hi_th = th.tensor(
                    self._action_center_hi
                ).to(action.device)
            norma = (
                action_hi - self._action_center_hi_th
            ) / self._action_scale_hi_th
            norma_next = (
                auto_next_action_hi - self._action_center_hi_th
            ) / self._action_scale_hi_th
            d = th.linalg.norm(norma, ord=2, dim=1, keepdim=True)
            d_next = th.linalg.norm(norma_next, ord=2, dim=1, keepdim=True)
            if self._reward_lo == 'distance':
                norma = (
                    auto_next_action_hi - self._action_center_hi_th
                ) / self._action_scale_hi_th
                reward_lo = -th.linalg.norm(norma, ord=2, dim=1, keepdim=True)
            elif self._reward_lo == 'potential':
                reward_lo = d - d_next
            elif self._reward_lo == 'potential2':
                reward_lo = d - self._gamma * d_next
            elif self._reward_lo == 'potential3':
                reward_lo = d - self._gamma * d_next
                if d_next < 0.1:
                    reward_lo += 1
            else:
                raise ValueError('Unknown low-level reward {self._reward_lo}')
        else:
            if self._reward_lo == 'distance':
                reward_lo = -th.linalg.norm(
                    auto_next_action_hi, ord=2, dim=1, keepdim=True
                )
            else:
                raise ValueError('Unknown low-level reward {self._reward_lo}')
        reward_lo = (
            reward_lo
            + fell_over * self._fallover_penalty_lo
            - self._ctrl_cost_lo * th.square(action_lo).sum(dim=1, keepdim=True)
        )

        d = dict(
            action_lo=action_lo,
            action_hi=action_hi,
            auto_next_action_hi=auto_next_action_hi,
            reward=reward,
            reward_lo=reward_lo,
            terminal=done,
            step=obs['time'].remainder(self._action_interval_hi).long(),
            fell_over=fell_over,
        )
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs[k]
            d[f'next_obs_{k}'] = next_obs[k]
        if self._gspace_key != 'observation':
            d[f'obs_{self._gspace_key}'] = obs[self._gspace_key]
            d[f'next_obs_{self._gspace_key}'] = next_obs[self._gspace_key]

        self._buffer.put_row(d)
        self._cur_rewards.append(reward)
        self._cur_rewards_lo.append(reward_lo)

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            self.update()
            self._cur_rewards.clear()
            self._cur_rewards_lo.clear()
            self._n_samples_since_update = 0

    def _update(self):
        self._update_lo()
        self._update_hi()

    def _update_lo(self):
        model = self._model.lo
        target = self._target.lo
        optim = self._optim.lo

        def act_logp(obs):
            dist = model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor_lo
            return action, log_prob

        for _ in range(self._num_updates):
            batch = self._buffer.get_batch(self._bsz)
            reward = batch['reward_lo']
            obs = {k: batch[f'obs_{k}'] for k in self._obs_lo_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_lo_keys}
            for k, m in self._obs_lo_mask.items():
                m = m.to(obs[k])
                obs[k] = obs[k] * m
                obs_p[k] = obs_p[k] * m
                self._obs_lo_mask[k] = m
            obs['desired_goal'] = batch['action_hi']
            obs_p['desired_goal'] = batch['auto_next_action_hi']
            not_fell_over = th.logical_not(batch['fell_over'])

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(target.q(q_in), dim=-1).values
                # Assume that low-level epsiodes don't end
                backup = reward + self._gamma * not_fell_over * (
                    q_tgt - self._log_alpha_lo.detach().exp() * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch['action_lo'], **obs)
            q = model.q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss
            optim.q.zero_grad()
            q_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.q.parameters(), self._clip_grad_norm
                )
            optim.q.step()

            # Policy update
            for param in model.q.parameters():
                param.requires_grad_(False)

            a, log_prob = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(model.q(q_in), dim=-1).values
            pi_loss = (self._log_alpha_lo.detach().exp() * log_prob - q).mean()
            optim.pi.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.pi.parameters(), self._clip_grad_norm
                )
            optim.pi.step()

            for param in model.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha_lo:
                alpha_loss = -(
                    self._log_alpha_lo.exp()
                    * (log_prob.mean().cpu() + self._target_entropy_lo).detach()
                )
                self._optim_alpha_lo.zero_grad()
                alpha_loss.backward()
                self._optim_alpha_lo.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(target.q.parameters(), model.q.parameters()):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('LossLo/Policy', pi_loss.item())
        self.tbw_add_scalar('LossLo/QValue', q_loss.item())
        self.tbw_add_scalar('HealthLo/Entropy', -log_prob.mean())
        if self._optim_alpha_lo:
            self.tbw_add_scalar(
                'HealthLo/Alpha', self._log_alpha_lo.exp().item()
            )
        self.tbw.add_scalars(
            'HealthLo/GradNorms',
            {
                k: v.grad.norm().item()
                for k, v in self._model.named_parameters()
                if v.grad is not None
            },
            self.n_samples,
        )

        avg_cr = th.cat(self._cur_rewards_lo).mean().item()
        log.info(
            f'Sample {self._n_samples} lo: up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {self._log_alpha_lo.exp().item():.03f}'
        )

    def _relabel_goal(
        self,
        batch: Dict[str, th.Tensor],
        mask: th.Tensor,
        obs_idx: th.Tensor,
        obs_p_idx,
    ) -> th.Tensor:
        bsz = self._bsz
        c = self._action_interval_hi

        # Query lo actions on a bsz*c batch so we can do it one go. Compute
        # the proper mask and padding beforehand, though.
        full_mask = mask.clone()
        ones = th.ones(
            (mask.shape[0], 1), dtype=th.bool, device=mask.device
        ).expand(mask.shape)
        for i in range(1, c):
            full_mask.scatter_(1, (obs_idx + i).unsqueeze(1), ones)

        def gather_at_fmask(x):
            return x.masked_select(full_mask.unsqueeze(2)).view(bsz, c, -1)

        # High-level action relabeling: propose some candidate goals,
        # compute intermediate ones with the transition function, and pick
        # the one with the highest log probability under the current
        # low-level policy.
        gf = self._goal_features
        gcand: List[th.Tensor] = []
        # Original action
        if len(gf) == 1:
            gcand.append(
                dim_select(batch['action_hi'], 1, obs_idx).unsqueeze(-1)
            )
        else:
            gcand.append(dim_select(batch['action_hi'], 1, obs_idx))
        # What actionally happend
        gobs = dim_select(
            batch[f'obs_{self._gspace_key}'][:, :, gf], 1, obs_idx
        )
        gobs_p = dim_select(
            batch[f'next_obs_{self._gspace_key}'][:, :, gf], 1, obs_p_idx
        )
        gcand.append(gobs_p - gobs)
        # Random candidates in the vicinity
        dist = Normal(loc=gcand[-1], scale=0.5 * self._action_scale_hi_th)
        for i in range(8):
            gcand.append(dist.sample())
        # Clip to actual action output range
        for i in range(1, len(gcand)):
            gcand[i] = th.min(
                th.max(gcand[i], self._gspace_min_th), self._gspace_max_th
            )
        gobs = gather_at_fmask(batch[f'obs_{self._gspace_key}'][:, :, gf])
        act_mask = th.zeros(
            (mask.shape[0], c), dtype=th.bool, device=mask.device
        )
        act_mask[:, 0] = True
        lengths = obs_p_idx - obs_idx
        for i in range(1, c):
            act_mask[:, i] = lengths >= i
        act_mask = act_mask.unsqueeze(0)

        logp_lo: List[th.Tensor] = []
        ncand = len(gcand)
        cand = th.stack(gcand).unsqueeze(2).repeat(1, 1, c, 1)
        # Apply transition function
        gobs = gobs.unsqueeze(0)
        for i in range(c - 1):
            cand.narrow(2, i + 1, 1).copy_(
                gobs.narrow(2, i, 1)
                + cand.narrow(2, i, 1)
                - gobs.narrow(2, i + 1, 1)
            )
        inp_lo = {
            k: gather_at_fmask(batch[f'obs_{k}']) for k in self._obs_lo_keys
        }
        for k, m in self._obs_lo_mask.items():
            inp_lo[k] = inp_lo[k] * m
        for k, v in inp_lo.items():
            inp_lo[k] = (
                v.unsqueeze(0).repeat(ncand, 1, 1, 1).view(ncand * bsz * c, -1)
            )
        inp_lo['desired_goal'] = cand.view(ncand * bsz * c, -1)
        with th.no_grad():
            dist = self._model.lo.pi(inp_lo)

        # Rather than using squared differences of actions, we maximize the
        # low-level actions' log probability directly. With SAC, we have access
        # to an actual distribution that we can query, whereas in TD3 we just
        # add random noise.
        action_lo = (
            gather_at_fmask(batch['action_lo'])
            .unsqueeze(0)
            .repeat(ncand, 1, 1, 1)
            .view(ncand * bsz * c, -1)
        )
        # Clamp action for numerical stability when inversing the tanh
        # transform (the caching trick doesn't work here because we're not
        # sampling the action first).
        limit = 1 - 1e-7
        action_lo = action_lo.clamp(-limit, limit)
        # Sum log probs across actions and (masked) time , and pick goals at
        # maxima
        log_prob = dist.log_prob(action_lo).sum(dim=-1)
        log_prob = (log_prob.view(ncand, bsz, c) * act_mask).sum(dim=-1)
        cand_max = log_prob.max(dim=0).indices
        action_hi = dim_select(cand[:, :, 0], 0, cand_max)
        return action_hi

    def _update_hi(self):
        model = self._model.hi
        target = self._target.hi
        optim = self._optim.hi

        if self._gspace_min_th is None:
            self._gspace_min_th = th.tensor(
                self._gspace_min, device=self._buffer.device
            )
            self._gspace_max_th = th.tensor(
                self._gspace_max, device=self._buffer.device
            )
        obs_keys = self._obs_keys

        def act_logp(obs):
            dist = model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = self.scale_action_hi(action)
            return action, log_prob

        bsz = self._bsz
        if self._dense_hi_updates:
            n = self._num_updates
        else:
            n = int(np.ceil(self._num_updates / self._action_interval_hi))
        it = 0
        while it < n:
            c = self._action_interval_hi
            k = c * 2 - 1
            batch = self._buffer.get_trajs(bsz, k)

            # Grab transitions from step 0 to self._action_interval_hi - 1 or
            # until a terminal state
            step = batch['step']
            acc = th.zeros_like(step)
            acc[:, 0] = step[:, 0] == 0
            for i in range(1, k):
                acc[:, i] = acc[:, i - 1] + (step[:, i] == 0)
            mask = acc == 1
            obs_idx = th.where(th.logical_and(step == 0, mask))[1]
            m_terminal = th.logical_and(batch['terminal'], mask)
            m_last_step = th.logical_and(step == c - 1, mask)
            obs_p_idx = th.where(th.logical_or(m_terminal, m_last_step))[1]
            if obs_idx.shape != obs_p_idx.shape:
                # We might run into this condition when continuing from a
                # checkpoint. Since environments will be reset, we might end up
                # with our c*2 - 1 not catching full high-level transitions.
                # This is quite a hotfix; another solution would be to use some
                # staging logic for transitions, but since this should happen
                # rarely let's just do this instead.
                it -= 1
                continue

            not_done = th.logical_not(
                dim_select(batch['terminal'], 1, obs_p_idx)
            )

            if self._relabel_goals:
                action_hi = self._relabel_goal(batch, mask, obs_idx, obs_p_idx)
            else:
                action_hi = dim_select(batch['action_hi'], 1, obs_idx)

            if self._dense_hi_updates:
                off = th.randint(
                    c, obs_idx.shape, device=obs_idx.device
                ).remainder(obs_p_idx - obs_idx + 1)
                obs_idx_off = obs_idx + off
                obs = {
                    k: dim_select(batch[f'obs_{k}'], 1, obs_idx_off)
                    for k in obs_keys
                }
                obs['time'] = dim_select(batch['step'], 1, obs_idx_off)
                obs_p = {
                    k: dim_select(batch[f'next_obs_{k}'], 1, obs_p_idx)
                    for k in obs_keys
                }
                obs_p['time'] = th.zeros_like(obs['time'])

                reward = dim_select(batch['reward'], 1, obs_idx_off)
                for i in range(1, c):
                    obs_idx_off_i = obs_idx_off
                    reward += (
                        self._gamma ** i
                        * dim_select(
                            batch['reward'], 1, (obs_idx_off_i).min(obs_p_idx)
                        )
                        * (obs_idx_off_i <= obs_p_idx)
                    )
                gamma = th.zeros_like(reward) + self._gamma
                gamma.pow_(obs_p_idx - obs_idx_off + 1)
            else:
                obs = {
                    k: dim_select(batch[f'obs_{k}'], 1, obs_idx)
                    for k in obs_keys
                }
                reward = (batch['reward'] * mask).sum(dim=1) / mask.sum(dim=1)
                gamma = self._gamma
                obs_p = {
                    k: dim_select(batch[f'next_obs_{k}'], 1, obs_p_idx)
                    for k in obs_keys
                }

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(target.q(q_in), dim=-1).values
                backup = reward + gamma * not_done * (
                    q_tgt - self._log_alpha_hi.detach().exp() * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=action_hi, **obs)
            q = model.q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss
            optim.q.zero_grad()
            q_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.q.parameters(), self._clip_grad_norm
                )
            optim.q.step()

            # Policy update
            for param in model.q.parameters():
                param.requires_grad_(False)

            if self._dense_hi_updates:
                # No time input for policy, and Q-functions are queried as if step
                # would be 0 (i.e. we would take an action)
                obs['time'] = obs['time'] * 0
            a, log_prob = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(model.q(q_in), dim=-1).values
            pi_loss = (self._log_alpha_hi.detach().exp() * log_prob - q).mean()
            optim.pi.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.pi.parameters(), self._clip_grad_norm
                )
            optim.pi.step()

            for param in model.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha_hi:
                alpha_loss = -(
                    self._log_alpha_hi.exp()
                    * (log_prob.mean().cpu() + self._target_entropy_hi).detach()
                )
                self._optim_alpha_hi.zero_grad()
                alpha_loss.backward()
                self._optim_alpha_hi.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(target.q.parameters(), model.q.parameters()):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

            it += 1

        # These are the stats for the last update
        self.tbw_add_scalar('LossHi/Policy', pi_loss.item())
        self.tbw_add_scalar('LossHi/QValue', q_loss.item())
        self.tbw_add_scalar('HealthHi/Entropy', -log_prob.mean())
        if self._optim_alpha_hi:
            self.tbw_add_scalar(
                'HealthHi/Alpha', self._log_alpha_hi.exp().item()
            )
        self.tbw.add_scalars(
            'HealthHi/GradNorms',
            {
                k: v.grad.norm().item()
                for k, v in self._model.named_parameters()
                if v.grad is not None
            },
            self.n_samples,
        )

        avg_cr = th.cat(self._cur_rewards).mean().item()
        log.info(
            f'Sample {self._n_samples} hi: up {self._n_updates*n}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {self._log_alpha_hi.exp().item():.03f}'
        )
