# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from collections import defaultdict
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from hucc import ReplayBuffer
from hucc.agents import Agent

log = logging.getLogger(__name__)


class SACMTAgent(Agent):
    '''
    Soft Actor-Critic agent, with modifications for multi-task training:
    - Require dictionary observations
    - Pass dictionaries to policy and Q functions
    - If a 'task' observation is present and alpha should be learned, learn it
      per task.
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

        self.role = 'all'
        self.learner_group = None
        self.bcast_barrier = None
        self._model = model
        self._optim = optim
        self._bsz = int(cfg.batch_size)
        self._gamma = float(cfg.gamma)
        self._polyak = float(cfg.polyak)
        self._rpbuf_size = int(cfg.rpbuf_size)
        self._samples_per_update = int(cfg.samples_per_update)
        self._num_updates = int(cfg.num_updates)
        self._ups = 0
        self._warmup_samples = int(cfg.warmup_samples)
        self._randexp_samples = int(cfg.randexp_samples)
        self._clip_grad_norm = float(cfg.clip_grad_norm)
        self._update_reachability = bool(cfg.update_reachability)

        self._target_entropy = (
            -np.prod(env.action_space.shape) * cfg.target_entropy_factor
        )
        # Optimize log(alpha) so that we'll always have a positive factor
        # We use a dictionary to maintain per-task log(alpha) values
        log_alpha = np.log(cfg.alpha)
        if cfg.optim_alpha is None:
            self._log_alpha: Dict[int, th.Tensor] = defaultdict(
                lambda: th.tensor(log_alpha, dtype=th.float32)
            )
            self._optim_alpha = None
        else:
            self._log_alpha = defaultdict(
                lambda: th.tensor(
                    log_alpha, dtype=th.float32, requires_grad=True
                )
            )
            self._optim_alpha: Dict[int, Optimizer] = {}
            self._cfg_optim_alpha = cfg.optim_alpha

        self._ignore_eoe = bool(cfg.ignore_eoe)

        rpbuf_device = cfg.rpbuf_device if cfg.rpbuf_device != 'auto' else None
        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs, device=rpbuf_device
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space = env.action_space
        self._action_factor = env.action_space.high[0]
        self._obs_space = env.observation_space
        self._obs_keys = list(self._obs_space.spaces.keys())
        # We don't feed the raw features of the goal space to the models,
        # but we save them in the replay buffer because they're useful for
        # estimate controllablity via models
        if 'gs_observation' in self._obs_keys:
            self._obs_keys.remove('gs_observation')
        self._per_task_alpha = cfg.per_task_alpha
        if self._per_task_alpha and not 'task' in self._obs_space.spaces:
            raise ValueError(
                'Per-task alpha requested, but no "task" observation in environment'
            )

        self._task_samples: Dict[int, int] = defaultdict(int)  # seen in step()
        self._sampled_tasks: Dict[int, int] = defaultdict(
            int
        )  # used in update()
        self._task_sample_pri: Dict[int, float] = defaultdict(lambda: 1.0)
        self._key_to_task: Dict[int, str] = {}
        self._avg_tderr: Dict[str, float] = defaultdict(float)
        self._avg_tderr_alpha = 0.01

        self.set_checkpoint_attr(
            '_model',
            '_target',
            '_optim',
            '_log_alpha',
            '_optim_alpha',
            '_task_samples',
            '_sampled_tasks',
            '_key_to_task',
            '_avg_tderr',
            '_ups',
        )

        assert isinstance(
            env.action_space, gym.spaces.Box
        ), f'SACMTAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), f'SACMTAgent requires a dictionary observation (but got {type(env.action_space)})'

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig) -> gym.Space:
        # "time" input is used for internal book-keeping only
        d = copy(env.observation_space.spaces)
        if 'gs_observation' in d:
            del d['gs_observation']
        return gym.spaces.Dict(d)

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        with th.no_grad():
            if self._n_samples < self._randexp_samples and self.training:
                action = th.stack(
                    [
                        th.from_numpy(self._action_space.sample())
                        for i in range(env.num_envs)
                    ]
                ).to([v for v in obs.values()][0].device)
            else:
                mobs = copy(obs)
                if 'gs_observation' in mobs:
                    del mobs['gs_observation']
                dist = self._model.pi(mobs)
                assert (
                    dist.has_rsample
                ), f'rsample() required for policy distribution'
                if self.training:
                    action = dist.sample() * self._action_factor
                else:
                    action = dist.mean * self._action_factor
        return action, None

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, done, info = result

        if self._ignore_eoe:
            done.fill_(False)
        else:
            # Ignore terminal state if we have a timeout or the episode does
            # actually continue.
            for i in range(len(info)):
                if (
                    'TimeLimit.truncated' in info[i]
                    or 'EpisodeContinues' in info[i]
                ):
                    done[i] = False
        start_state = [False] * len(info)
        for i in range(len(info)):
            if info[i].get('time', -1) == 0:
                start_state[i] = True
        reached_goal = [0.0] * len(info)
        for i in range(len(info)):
            if info[i].get('reached_goal', False):
                reached_goal[i] = 1.0
        last_step_of_task = [False] * len(info)
        for i in range(len(info)):
            if info[i].get('LastStepOfTask', False) and not info[i].get(
                'RandomReset', False
            ):
                last_step_of_task[i] = True

        d = dict(
            action=action,
            reward=reward,
            terminal=done,
            start_state=th.tensor(start_state, device=done.device),
            reached_goal=th.tensor(reached_goal, device=done.device),
            last_step_of_task=th.tensor(last_step_of_task, device=done.device),
        )
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs[k]
            d[f'next_obs_{k}'] = next_obs[k]
        if 'gs_observation' in obs:
            d['gs_observation'] = obs['gs_observation']
        if 'task' in obs:
            key_idx = self._task_key_idx(obs['task'])
            # Store priorites for task, and also their keys so we can easily
            # update them later.
            # XXX Crossing my fingers that keys will fit into 64 bits.
            task_pri = th.zeros(done.shape, dtype=th.float)
            task_key = th.zeros(done.shape, dtype=th.long)
            for key, idx in key_idx.items():
                task_pri[idx] = self._task_sample_pri[key]
                task_key[idx] = key
            d['task_key'] = task_key

            # Keep track of observed sample count per task while we're at it.
            for key, idx in key_idx.items():
                self._task_samples[key] += len(idx)
                if not key in self._key_to_task:
                    srepr = ','.join(
                        [
                            str(v)
                            for v in th.where(obs['task'][idx][0] > 0)[0]
                            .cpu()
                            .numpy()
                        ]
                    )
                    self._key_to_task[key] = srepr

        self._buffer.put_row(d)
        self._cur_rewards.append(reward)

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        if self.role == 'learner':
            log.debug(f'rpbuf size {self._buffer.size}/{self._warmup_samples}')
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            if self.role == 'actor':
                # Skip initial update so that we lag one update behind the
                # learner
                if self._n_updates > 0:
                    log.debug('Actor waiting for new params')
                    self.bcast_barrier.wait()
                    for p in self._model.parameters():
                        dist.broadcast(p, src=0)
                    for p in self._target.parameters():
                        dist.broadcast(p, src=0)
                    log.debug('Received')
                self._n_updates += 1
            else:
                self.update()
                if self.role == 'learner':
                    log.debug('Learned bcast new params')
                    self.bcast_barrier.wait()
                    for p in self._model.parameters():
                        dist.broadcast(p, src=0)
                    for p in self._target.parameters():
                        dist.broadcast(p, src=0)
                    log.debug('Okey-dokey')
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def avg_tderr_per_task(self) -> Dict[str, float]:
        return self._avg_tderr

    def _task_key_idx(self, tasks):
        # Returns a dict of key: index for selecting the correct alpha
        # values for each task.
        utasks, tpos = th.unique(
            tasks, dim=0, return_inverse=True, sorted=False
        )
        utasks = utasks.cpu().numpy()
        ntasks = utasks.shape[0]
        keys = [hash(utasks[i].tobytes()) for i in range(ntasks)]
        idxs = [[] for i in range(ntasks)]
        for i, j in enumerate(tpos.cpu().numpy()):
            idxs[j].append(i)
        return dict(zip(keys, idxs))

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

        def task_map(task_keys):
            # Maps task keys in the batch to indices
            keys, pos = th.unique(task_keys, return_inverse=True, sorted=False)
            keys = keys.cpu().numpy()
            pos = pos.cpu().numpy()
            key_idx = defaultdict(list)
            for i, j in enumerate(pos):
                key_idx[keys[j]].append(i)
            return key_idx

        def alpha_for_tasks(task_keys):
            key_idx = task_map(task_keys)
            alpha = th.zeros(task_keys.shape[0], dtype=th.float32)
            for key, idx in key_idx.items():
                alpha[idx] = self._log_alpha[key].detach().exp()
            return alpha.to(task_keys.device), key_idx

        task_keys: List[th.Tensor] = []
        for _ in range(self._num_updates):
            self._ups += 1
            batch = self._buffer.get_batch(self._bsz, device=mdevice)
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            reward = batch['reward']
            not_done = th.logical_not(batch['terminal'])

            # Determine task counts in this batch
            if self._per_task_alpha:
                with th.no_grad():
                    task_alpha, task_key_idx = alpha_for_tasks(
                        batch['task_key']
                    )
            if 'task_key' in batch:
                task_keys.append(batch['task_key'])

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)

                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._target.q(q_in), dim=-1).values
                if not self._per_task_alpha:
                    alpha = self._log_alpha['_'].detach().exp()
                else:
                    alpha = task_alpha
                backup = reward + self._gamma * not_done * (
                    q_tgt - alpha * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch['action'], **obs)
            q = self._model.q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup, reduction='none')
            q2_loss = F.mse_loss(q2, backup, reduction='none')
            self._optim.q.zero_grad()
            q_loss = q1_loss.mean() + q2_loss.mean()
            q_loss.backward()
            if self.learner_group:
                for p in self._model.q.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, group=self.learner_group)
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
            q = th.min(self._model.q(q_in), dim=-1).values
            if not self._per_task_alpha:
                alpha = self._log_alpha['_'].detach().exp()
            else:
                alpha = task_alpha.detach()
            pi_loss = alpha * log_prob - q
            pi_loss = pi_loss.mean()
            self._optim.pi.zero_grad()
            pi_loss.backward()
            if self.learner_group:
                for p in self._model.pi.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, group=self.learner_group)
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.pi.parameters(), self._clip_grad_norm
                )
            self._optim.pi.step()

            for param in self._model.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha is not None:

                def optim_alpha(key):
                    if not key in self._optim_alpha:
                        self._optim_alpha[key] = hydra.utils.instantiate(
                            self._cfg_optim_alpha, [self._log_alpha[key]]
                        )
                    return self._optim_alpha[key]

                if not self._per_task_alpha:
                    # This is slight reording of the formulation in
                    # https://github.com/rail-berkeley/softlearning, mostly so we
                    # don't need to create temporary tensors. log_prob is the only
                    # non-scalar tensor, so we can compute its mean first.
                    alpha_loss = -(
                        self._log_alpha['_'].exp()
                        * (
                            log_prob.mean().cpu() + self._target_entropy
                        ).detach()
                    )
                    optim_alpha('_').zero_grad()
                    alpha_loss.backward()
                    optim_alpha('_').step()
                else:
                    for key, idx in task_key_idx.items():
                        alpha_loss = -(
                            self._log_alpha[key].exp()
                            * (
                                log_prob[idx].mean().cpu()
                                + self._target_entropy
                            ).detach()
                        )
                        optim_alpha(key).zero_grad()
                        alpha_loss.backward()
                        optim_alpha(key).step()

            # Update reachability network via TD learning
            if self._update_reachability and hasattr(
                self._model, 'reachability'
            ):
                with th.no_grad():
                    a_p, log_prob_p = act_logp(obs_p)
                    r_in = dict(action=a_p, **obs_p)
                    r_tgt = self._target.reachability(r_in).view(-1)
                    propagate = th.logical_not(batch['last_step_of_task'])
                    r_backup = batch['reached_goal'] + propagate * r_tgt

                r_in = dict(action=batch['action'], **obs)
                r_est = self._model.reachability(r_in).view(-1)
                r_loss = F.mse_loss(r_est, r_backup, reduction='mean')
                self._optim.reachability.zero_grad()
                r_loss.backward()
                if self.learner_group:
                    for p in self._model.reachability.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, group=self.learner_group)
                if self._clip_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        self._model.reachability.parameters(),
                        self._clip_grad_norm,
                    )
                self._optim.reachability.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.parameters(), self._model.parameters()
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        if task_keys:
            task_keys_c = th.cat(task_keys)
            tasks, counts = th.unique(
                task_keys_c, return_counts=True, sorted=False
            )
            for t, c in zip(tasks.cpu().numpy(), counts.cpu().numpy()):
                self._sampled_tasks[t] += c

        # These are the stats for the last update
        with th.no_grad():
            mean_alpha = np.mean(
                [a.exp().item() for a in self._log_alpha.values()]
            )
        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        if self._update_reachability and hasattr(self._model, 'reachability'):
            self.tbw_add_scalar('Loss/Reachability', r_loss.item())
        self.tbw_add_scalar('Health/Entropy', -log_prob.mean())
        if self._optim_alpha:
            if not self._per_task_alpha:
                self.tbw_add_scalar(
                    'Health/Alpha', self._log_alpha['_'].exp().item()
                )
            else:
                self.tbw_add_scalar('Health/MeanAlpha', mean_alpha)
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
            for i in range(a.shape[1]):
                self.tbw.add_histogram(
                    f'Health/PolicyA{i}', a[i][:100], self.n_samples
                )
            self.tbw.add_histogram('Health/Q1', q1[:100], self.n_samples)
            self.tbw.add_histogram('Health/Q2', q2[:100], self.n_samples)

        # Log TD errors per abstraction
        if 'task_key' in batch and self._n_updates % 10 == 1:
            task_key_idx = task_map(task_keys[-1])
            with th.no_grad():
                tderrs = {}
                for key, idx in task_key_idx.items():
                    task = self._key_to_task[key]
                    tde1 = q1_loss[idx].sqrt().mean().item()
                    tde2 = q1_loss[idx].sqrt().mean().item()
                    tderrs[task] = (tde1 + tde2) / 2
                self.tbw.add_scalars(
                    'Health/AbsTDErrorMean', tderrs, self._n_samples
                )
                for task, err in tderrs.items():
                    self._avg_tderr[task] *= 1.0 - self._avg_tderr_alpha
                    self._avg_tderr[task] += self._avg_tderr_alpha * err

        self.tbw.add_scalars(
            'Agent/SampledTasks',
            {self._key_to_task[k]: v for k, v in self._sampled_tasks.items()},
            self._n_samples,
        )
        self.tbw.add_scalars(
            'Agent/SamplesPerTask',
            {self._key_to_task[k]: v for k, v in self._task_samples.items()},
            self._n_samples,
        )

        avg_cr = th.cat(self._cur_rewards).mean().item()
        if self._update_reachability and hasattr(self._model, 'reachability'):
            log.info(
                f'Sample {self._n_samples}, up {self._ups}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, r loss {r_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {mean_alpha:.03f}'
            )
        else:
            log.info(
                f'Sample {self._n_samples}, up {self._ups}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.mean().item():+.03f}, alpha {mean_alpha:.03f}'
            )
