# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union

import gym
import numpy as np
import torch as th
import torch.distributions as D
from torch.nn import functional as F


class Bandit:
    def __init__(self, action_space: gym.Space):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f'Bandit needs a discrete action space (but got {type(action_space)})'
            )
        n = action_space.n
        self.q = th.zeros(n)
        self.n = th.zeros(n)

    def update(self, action, r):
        self.n[action] += 1
        w = 1.0 / self.n[action]
        self.q[action] = (1 - w) * self.q[action] + w * r

    def sample_action(self):
        return th.argmax(th.rand(self.q.shape) * (self.q == self.q.max()))

    def best_action(self):
        return th.argmax(self.q)

    def dist(self):
        # Distribution over arms according to estimated returns -- mostly for
        # debugging purposes
        return D.Categorical(probs=F.softmax(self.q, dim=0))


class UniformBandit(Bandit):
    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)

    def sample_action(self):
        return th.randint(0, self.q.shape[0], (1,))[0]


class EpsGreedyBandit(Bandit):
    def __init__(self, action_space: gym.Space, eps: float = 0.1):
        super().__init__(action_space)
        self.eps = eps

    def sample_action(self):
        if th.rand(1)[0] < self.eps:
            return th.randint(0, self.q.shape[0], (1,))[0]
        return th.argmax(th.rand(self.q.shape) * (self.q == self.q.max()))


class UCBBandit(Bandit):
    def __init__(
        self, action_space: gym.Space, beta: Union[float, str] = 'auto'
    ):
        super().__init__(action_space)
        if beta == 'auto':
            self.beta = 0.0
            self.auto_beta = True
        else:
            self.beta = float(beta)
            self.auto_beta = False

    def update(self, action, r):
        if self.auto_beta:
            self.beta = max(self.beta, r)
        super().update(action, r)

    def sample_action(self):
        if self.n.sum() < self.q.shape[0]:
            return self.n.sum().long()
        c = th.sqrt(self.n.sum().log() / self.n)
        return th.argmax(self.q + self.beta * c)


class DiscountedUCBBandit(Bandit):
    def __init__(
        self,
        action_space: gym.Space,
        gamma: float = 0.9,
        beta: Union[str, float] = 'auto',
    ):
        super().__init__(action_space)
        self.gamma = gamma
        if beta == 'auto':
            self.beta = 0.0
            self.auto_beta = True
        else:
            self.beta = float(beta)
            self.auto_beta = False

    def update(self, a, r):
        if self.auto_beta:
            self.beta = max(self.beta, r)

        self.q *= self.gamma
        self.q[a] += r
        self.n *= self.gamma
        self.n[a] += 1

    def sample_action(self):
        if self.n.sum() < self.q.shape[0]:
            return self.n.sum().long()
        c = th.sqrt(self.n.sum().log() / self.n)
        return th.argmax((self.q / (self.n + 1)) + 2 * self.beta * c)

    def best_action(self):
        return th.argmax(self.q / (self.n + 1))

    def dist(self):
        # Distribution over arms according to estimated returns -- mostly for
        # debugging purposes
        return D.Categorical(probs=F.softmax(self.q / (self.n + 1e-7), dim=0))


class SlidingWindowUCBBandit(Bandit):
    def __init__(
        self,
        action_space: gym.Space,
        window: int = 10,
        beta: Union[str, float] = 0,
    ):
        super().__init__(action_space)
        self.rbuf = th.zeros(window, action_space.n)
        self.abuf = th.zeros(window, action_space.n)
        self.window = window
        self.bidx = 0
        self.step = 0

        if beta == 'auto':
            self.beta = 0.0
            self.auto_beta = True
        else:
            self.beta = float(beta)
            self.auto_beta = False

    def update(self, a, r):
        if self.auto_beta:
            self.beta = max(self.beta, r)

        self.rbuf[self.bidx].fill_(0)
        self.rbuf[self.bidx][a] = r
        self.abuf[self.bidx].fill_(0)
        self.abuf[self.bidx][a] = 1
        self.bidx = (self.bidx + 1) % self.window
        self.step += 1

    def sample_action(self):
        if self.step < self.q.shape[0]:
            return th.tensor(self.step, dtype=th.long)
        mu = self.rbuf.sum(dim=0) / self.abuf.sum(dim=0)
        c = th.sqrt(np.log(min(self.step, self.window)) / self.abuf.sum(dim=0))
        return th.argmax(mu + self.beta * c)

    def best_action(self):
        return th.argmax(self.rbuf.sum(0) / (self.abuf.sum(0) + 1))

    def dist(self):
        return D.Categorical(
            probs=F.softmax(self.rbuf.sum(0) / (self.abuf.sum(0) + 1e-7), dim=0)
        )


class Agent57Bandit(SlidingWindowUCBBandit):
    '''
    Bandit algorithm used in Agent57. Essentially a mix between sliding window
    UCB and epsilon-greedy.
    '''

    def __init__(
        self,
        action_space: gym.Space,
        window: int = 90,
        beta: float = 1.0,
        eps: float = 0.5,
    ):
        super().__init__(action_space, window, beta)
        self.eps = eps

    def sample_action(self):
        if self.step < self.q.shape[0]:
            return th.tensor(self.step, dtype=th.long)
        if th.rand(1)[0] < self.eps:
            return th.randint(0, self.q.shape[0], (1,))[0]
        mu = self.rbuf.sum(dim=0) / self.abuf.sum(dim=0)
        c = th.sqrt(1 / self.abuf.sum(dim=0))
        return th.argmax(mu + self.beta * c)


class ThompsonBandit(Bandit):
    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)
        n = action_space.n
        self.tau = th.zeros(n) + 1e-3
        self.mu = th.ones(n)

    def update(self, a, r):
        super().update(a, r)
        self.mu[a] = ((self.tau[a] * self.mu[a]) + (self.n[a] * self.q[a])) / (
            self.tau[a] + self.n[a]
        )
        self.tau[a] += 1

    def sample_action(self):
        dist = (th.randn(self.q.shape) / self.tau.sqrt()) + self.mu
        return th.argmax(dist)

    def dist(self):
        dist = (th.randn(self.q.shape) / self.tau.sqrt()) + self.mu
        return D.Categorical(probs=F.softmax(dist, dim=0))
