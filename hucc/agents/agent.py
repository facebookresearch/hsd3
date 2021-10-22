# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch as th
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from hucc.agents.utils import batch_2to1

log = logging.getLogger(__name__)


class Agent:
    '''
    Minimal interface for agents.
    '''

    def __init__(self, cfg: DictConfig):
        self._tbw: Optional[SummaryWriter] = None
        self._n_updates = 0
        self._n_samples = 0
        # XXX we rely on step() implementations updating this counter...
        self._n_steps = 0
        self._training = True
        self._checkpoint_attr: List[str] = []

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        return env.observation_space

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        return env.action_space

    @property
    def tbw(self) -> Optional[SummaryWriter]:
        return self._tbw

    @tbw.setter
    def tbw(self, writer: SummaryWriter):
        self._tbw = writer

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, training) -> None:
        self._training = training
        if hasattr(self, '_model'):  # XXX
            model = getattr(self, '_model')
            if training:
                model.train()
            else:
                model.eval()

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        raise NotImplementedError()

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[Any, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        '''
        Indended to be called after env.step() for samples that should be used
        for training.
        - env: The env instance
        - obs: Previous observatio used to determine the action
        - action: Action that was taken
        - extra: The extra value returned from the previous action() call
        - result: Return value of env.step(), i.e. (next_obs, reward, done,
          info).
        '''
        raise NotImplementedError()

    def set_checkpoint_attr(self, *attr: str) -> None:
        for a in attr:
            assert hasattr(self, a), f'Missing checkpoint attribute {a}'
        self._checkpoint_attr = list(attr)

    def load_checkpoint(self, fd) -> None:
        data = th.load(fd, map_location='cpu')
        self._n_updates = data['_n_updates']
        self._n_steps = data['_n_steps']
        self._n_samples = data['_n_samples']
        training = data['_training']

        def load(val, data):
            if isinstance(val, nn.Module) or isinstance(val, optim.Optimizer):
                val.load_state_dict(data)
            elif isinstance(val, SimpleNamespace):
                for k, v in val.__dict__.items():
                    load(v, data[k])
            elif isinstance(val, defaultdict):
                val.clear()
                for k, v in data.items():
                    val[k] = v
            elif isinstance(val, th.Tensor):
                val.detach().copy_(data)
            else:
                return False
            return True

        for a in self._checkpoint_attr:
            val = getattr(self, a)
            if not load(val, data[a]):
                setattr(self, a, data[a])
        self.training = training

    def save_checkpoint(self, fd) -> None:
        data: Dict[str, Any] = dict()
        data['_n_updates'] = self._n_updates
        data['_n_samples'] = self._n_samples
        data['_n_steps'] = self._n_steps
        data['_training'] = self._training

        def store(val):
            if isinstance(val, nn.Module) or isinstance(val, optim.Optimizer):
                return val.state_dict()
            elif isinstance(val, SimpleNamespace):
                return {k: store(v) for k, v in val.__dict__.items()}
            elif isinstance(val, defaultdict):
                return dict(val)
            return val

        for a in self._checkpoint_attr:
            val = getattr(self, a)
            data[a] = store(val)
        th.save(data, fd)

    def update(self) -> None:
        self._n_updates += 1
        self._update()

    def tbw_add_scalars(
        self,
        title: str,
        vals: th.Tensor,
        agg=['mean', 'min', 'max'],
        n_samples=None,
    ):
        if self.tbw is None:
            return
        data = {a: getattr(vals, a)() for a in agg}
        self.tbw.add_scalars(
            title, data, n_samples if n_samples is not None else self._n_samples
        )

    def tbw_add_scalar(self, title: str, value: float, n_samples=None):
        if self.tbw is None:
            return
        self.tbw.add_scalar(
            title,
            value,
            n_samples if n_samples is not None else self._n_samples,
        )

    def _update(self) -> None:
        raise NotImplementedError()

    def _update_v(
        self,
        model: nn.Module,
        optim: SimpleNamespace,
        obs: th.Tensor,
        ret: th.Tensor,
        train_v_iters: int,
        max_grad_norm: float = math.inf,
    ) -> float:
        '''
        Common value function update.
        '''
        # Assume non-recurrent model and hence flatten observations and returns
        # into a single batch dimension
        obs, ret = batch_2to1(obs), batch_2to1(ret)

        if hasattr(model.v, 'fit'):
            # Custom baseline update
            model.v.fit(obs, ret)
            with th.no_grad():
                return F.mse_loss(model.v(obs), ret).item()

        ret = ret.detach()
        for i in range(train_v_iters):
            optim.v.zero_grad()
            value = model.v(obs).view_as(ret)
            loss = F.mse_loss(value, ret)
            loss.backward()
            if max_grad_norm != math.inf:
                nn.utils.clip_grad_norm_(model.v.parameters(), max_grad_norm)
            optim.v.step()
        return loss.item()
