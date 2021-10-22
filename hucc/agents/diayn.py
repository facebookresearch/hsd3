# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import math
from copy import deepcopy
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

log = logging.getLogger(__name__)


class DIAYNAgent(Agent):
    '''
    Diversity is All You Need with Soft-Actor Critic.
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
        if not hasattr(model, 'phi'):
            raise ValueError('Model needs "phi" module')

        self._model = model
        self._optim = optim
        self._n_skills = int(cfg.n_skills)
        self._bsz = int(cfg.batch_size)
        self._gamma = float(cfg.gamma)
        self._polyak = float(cfg.polyak)
        self._rpbuf_size = int(cfg.rpbuf_size)
        self._samples_per_update = int(cfg.samples_per_update)
        self._num_updates = int(cfg.num_updates)
        self._warmup_samples = int(cfg.warmup_samples)
        self._phi_obs = str(cfg.phi_obs)
        self._phi_obs_feats: Optional[List[int]] = None
        if cfg.phi_obs_feats is not None:
            self._phi_obs_feats = list(
                map(int, str(cfg.phi_obs_feats).split('#'))
            )
        elif cfg.phi_obs_n > 0:
            self._phi_obs_feats = list(range(cfg.phi_obs_n))
        self._add_p_z = bool(cfg.add_p_z)

        self._target_entropy = -np.prod(env.action_space.shape)
        # Optimize log(alpha) so that we'll always have a positive factor
        log_alpha = np.log(cfg.alpha)
        if cfg.optim_alpha is None:
            self._log_alpha = th.tensor(log_alpha)
            self._optim_alpha = None
        else:
            self._log_alpha = th.tensor(log_alpha, requires_grad=True)
            self._optim_alpha = hydra.utils.instantiate(
                cfg.optim_alpha, [self._log_alpha]
            )

        self._log_p_z = th.tensor(1 / self._n_skills).log().to(env.device)

        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs
        )
        self._n_samples_since_update = 0

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space = env.action_space
        self._action_factor = env.action_space.high[0]

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha',
            '_optim_alpha',
        )

        assert isinstance(
            env.action_space, gym.spaces.Box
        ), f'DIAYNAgent requires a continuous (Box) action space (but got {type(env.action_space)})'

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig) -> gym.Space:
        obs_space = env.observation_space['observation']
        n_skills = int(cfg.n_skills)
        # Add one-hot encoding of current skill to all observations
        return gym.spaces.Box(
            low=np.concatenate([obs_space.low, np.zeros(n_skills)]),
            high=np.concatenate([obs_space.high, np.ones(n_skills)]),
        )

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        with th.no_grad():
            prev_z = env.ctx.get(
                'z', th.zeros((env.num_envs,), dtype=th.long, device=env.device)
            )
            # Sample a new skill at the beginning of an episode
            sample_z = (obs['time'] == 0).squeeze(1)
            if sample_z.any():
                if self.training:
                    sample = th.randint(
                        0,
                        self._n_skills,
                        size=(env.num_envs,),
                        device=env.device,
                    )
                else:
                    # For evaluation, just enumerate all skills
                    sample = th.cat(
                        [th.arange(0, self._n_skills)]
                        * math.ceil(env.num_envs / self._n_skills)
                    )[: env.num_envs].to(env.device)
                z = (th.logical_not(sample_z)) * prev_z + (sample_z) * sample
            else:
                z = prev_z
            env.ctx['z'] = z

            aug_obs = th.cat(
                [obs['observation'], F.one_hot(z, self._n_skills).float()],
                dim=-1,
            )
            dist = self._model.pi(aug_obs)
            assert (
                dist.has_rsample
            ), f'rsample() required for policy distribution'
            if self.training:
                action = dist.sample() * self._action_factor
            else:
                action = dist.mean * self._action_factor
        return action, {'z': z}

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        oobs = obs['observation']
        z = extra['z']
        next_obs, reward, done, info = result
        next_oobs = next_obs['observation']

        # Ignore terminal state if we have a timeout
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                done[i] = False

        batch = dict(
            obs=oobs,
            z=z,
            next_obs=next_oobs,
            action=action.squeeze(-1),
            terminal=done,
        )
        if self._phi_obs != 'observation':
            batch['phi_obs'] = obs[self._phi_obs]
        self._buffer.put_row(batch)

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            self.update()
            self._n_samples_since_update = 0

    def _update(self):
        def act_logp(obs):
            dist = self._model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor
            return action, log_prob

        rewards = []
        for _ in range(self._num_updates):
            batch = self._buffer.get_batch(self._bsz)
            # Ensure that action has a batch dimension
            action = batch['action'].view(batch['obs'].shape[0], -1)
            z = batch['z']
            z_one_hot = F.one_hot(z, self._n_skills).float()
            phi_obs = batch.get('phi_obs', batch['obs'])
            if self._phi_obs_feats is not None:
                phi_obs = phi_obs[:, self._phi_obs_feats]
            not_done = th.logical_not(batch['terminal'])

            # Compute pseudo-reward with discriminator
            with th.no_grad():
                reward = -F.cross_entropy(
                    self._model.phi(phi_obs), z, reduction='none'
                )
                # Subtract baseline
                if self._add_p_z:
                    reward = reward - self._log_p_z
                rewards.append(reward.mean().item())

            # Backup for Q-Function
            with th.no_grad():
                obs_p = th.cat([batch['next_obs'], z_one_hot], dim=1)
                a_p, log_prob_p = act_logp(obs_p)

                q_in = th.cat([obs_p, a_p], dim=1)
                q_tgt = th.min(self._target.q(q_in), dim=-1).values
                backup = reward + self._gamma * not_done * (
                    q_tgt - self._log_alpha.detach().exp() * log_prob_p
                )

            # Q-Function update
            obs = th.cat([batch['obs'], z_one_hot], dim=1)
            q_in = th.cat([obs, action], dim=1)
            q = self._model.q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss
            self._optim.q.zero_grad()
            q_loss.backward()
            self._optim.q.step()

            # Policy update
            for param in self._model.q.parameters():
                param.requires_grad_(False)

            a, log_prob = act_logp(obs)
            q_in = th.cat([obs, a], dim=1)
            q = th.min(self._model.q(q_in), dim=-1).values
            pi_loss = (self._log_alpha.detach().exp() * log_prob - q).mean()
            self._optim.pi.zero_grad()
            pi_loss.backward()
            self._optim.pi.step()

            for param in self._model.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha:
                # This is slight reording of the formulation in
                # https://github.com/rail-berkeley/softlearning, mostly so we
                # don't need to create temporary tensors. log_prob is the only
                # non-scalar tensor, so we can compute its mean first.
                alpha_loss = -(
                    self._log_alpha.exp()
                    * (log_prob.mean().cpu() + self._target_entropy).detach()
                )
                self._optim_alpha.zero_grad()
                alpha_loss.backward()
                self._optim_alpha.step()

            # Update discriminator
            self._optim.phi.zero_grad()
            phi_loss = F.cross_entropy(self._model.phi(phi_obs), z)
            phi_loss.backward()
            self._optim.phi.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.q.parameters(), self._model.q.parameters()
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        self.tbw_add_scalar('Loss/Discriminator', phi_loss.item())
        self.tbw_add_scalar('Avg Reward', np.mean(rewards))
        self.tbw_add_scalar('Health/Entropy', log_prob.mean().item())
        if self._optim_alpha:
            self.tbw_add_scalar('Health/Alpha', self._log_alpha.exp().item())

        msg = log.debug
        if (self._n_updates * self._num_updates) % 50 == 0:
            msg = log.info
        msg(
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, phi loss {phi_loss.item():+.03f}, avg reward {np.mean(rewards):+.03}, alpha {self._log_alpha.exp().item():.03f}'
        )
