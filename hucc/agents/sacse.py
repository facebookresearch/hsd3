# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.models import TracedModule

log = logging.getLogger(__name__)


class SACSEAgent(Agent):
    '''
    Soft Actor-Critic "Switching Ensemble" agent, originally proposed with TD3
    in "Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?",
    Nachum et al., 2019.
    Both `module.pi` and `module.q` are will be conditioned on an extra one-hot
    'task' input, corresponding to the ensemble index. This is because we have
    some models bilinear models available to work with that :)
    '''

    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        if not hasattr(model, 'pi'):
            raise ValueError('Model needs "pi" module')
        if not hasattr(model, 'q'):
            raise ValueError('Model needs "q" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'SACSEAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )

        self._model = model
        self._n_models = int(cfg.n_models)
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
        self._c_switch = int(cfg.c_switch)

        self._target_entropy = (
            -np.prod(env.action_space.shape) * cfg.target_entropy_factor
        )
        # Optimize log(alpha) so that we'll always have a positive factor
        log_alpha = self._n_models * [np.log(cfg.alpha)]
        if cfg.optim_alpha is None:
            self._log_alpha = th.tensor(log_alpha)
            self._optim_alpha = None
        else:
            self._log_alpha = th.tensor(log_alpha, requires_grad=True)
            self._optim_alpha = hydra.utils.instantiate(
                cfg.optim_alpha, [self._log_alpha]
            )

        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._q = self._model.q
        self._q_tgt = self._target.q
        if cfg.trace:
            self._q = TracedModule(self._model.q)
            self._q_tgt = TracedModule(self._target.q)

        self._action_space = env.action_space
        self._action_factor = env.action_space.high[0]
        self._obs_space = self.effective_observation_space(env, cfg)
        self._obs_keys = list(self._obs_space.spaces.keys())

        self._mod: Dict[str, nn.Module] = {}

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha',
            '_optim_alpha',
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'SACSEAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        spaces = copy(env.observation_space.spaces)
        if not 'time' in spaces:
            raise ValueError(f'SACSEAgent requires a "time" observation')
        del spaces['time']
        if 'task' in spaces:
            raise ValueError(
                f'SACSEAgent can\'t work with a "task" observation'
            )
        spaces['task'] = gym.spaces.Box(
            low=0, high=1, shape=(cfg.n_models,), dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        step = obs['time'].remainder(self._c_switch).long().view(-1)
        keep_idx = step != 0
        idx = env.ctx.get('idx', None)
        if self.training:
            new_idx = th.randint(
                0, self._n_models, (env.num_envs,), device=step.device
            )
        else:
            # Fixed ensemble member for evaluations
            new_idx = th.zeros_like(step)
        if idx is None:
            idx = new_idx
        else:
            idx = keep_idx * idx + th.logical_not(keep_idx) * new_idx
        mobs = copy(obs)
        del mobs['time']
        mobs['task'] = F.one_hot(idx, self._n_models).float()

        with th.no_grad():
            if self._n_samples < self._randexp_samples and self.training:
                action = th.stack(
                    [
                        th.from_numpy(self._action_space.sample())
                        for i in range(env.num_envs)
                    ]
                ).to(list(self._model.parameters())[0].device)
            else:
                dist = self._model.pi(mobs)
                assert (
                    dist.has_rsample
                ), f'rsample() required for policy distribution'
                if self.training:
                    action = dist.sample() * self._action_factor
                else:
                    action = dist.mean * self._action_factor

        env.ctx['idx'] = idx
        return action, idx

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, done, info = result
        idx = extra
        # Ignore terminal state if we have a timeout
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                done[i] = False

        d = dict(
            action=action,
            reward=reward,
            terminal=done,
        )
        for k in self._obs_keys:
            if k != 'task':
                d[f'obs_{k}'] = obs[k]
                d[f'next_obs_{k}'] = next_obs[k]
        d['obs_task'] = idx
        d['next_obs_task'] = idx

        self._buffer.put_row(d)
        self._cur_rewards.append(reward)

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _update(self):
        for p in self._model.parameters():
            mdevice = p.device
            break

        def act_logp(obs):
            dist = self._model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor
            return action, log_prob

        # We'll equally feed samples to all models
        idx = th.arange(self._bsz * self._n_models) // self._bsz
        idx_in = F.one_hot(idx, self._n_models).to(
            dtype=th.float32, device=mdevice
        )

        for _ in range(self._num_updates):
            batch = self._buffer.get_batch(
                self._bsz * self._n_models,
                device=mdevice,
            )
            reward = batch['reward']
            not_done = th.logical_not(batch['terminal'])
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            obs['task'] = idx_in
            obs_p['task'] = idx_in
            alpha = (
                self._log_alpha.detach()
                .exp()
                .to(dtype=th.float32, device=mdevice)[idx]
            )

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._q_tgt(q_in), dim=-1).values
                backup = reward + self._gamma * not_done * (
                    q_tgt - alpha * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch['action'], **obs)
            q = self._q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss
            self._optim.q.zero_grad()
            q_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.q.parameters(), self._clip_grad_norm
                )
            self._optim.q.step()

            # Policy update
            for param in self._model.q.parameters():
                param.requires_grad_(False)

            a, log_prob = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(self._q(q_in), dim=-1).values
            pi_loss = (alpha * log_prob - q).mean()
            self._optim.pi.zero_grad()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.pi[0].parameters(), self._clip_grad_norm
                )
            self._optim.pi.step()

            for param in self._model.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha:
                alpha_loss = -(
                    self._log_alpha.exp()[idx]
                    * (log_prob.cpu() + self._target_entropy).detach()
                )
                self._optim_alpha.zero_grad()
                alpha_loss.mean().backward()
                self._optim_alpha.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.q.parameters(), self._model.q.parameters()
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        self.tbw_add_scalar('Health/Entropy', -log_prob.mean())
        if self._optim_alpha:
            self.tbw_add_scalar(
                'Health/Alpha', self._log_alpha.exp().mean().item()
            )
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
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {self._log_alpha.exp().mean().item():.03f}'
        )
