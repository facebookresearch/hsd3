# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from copy import copy, deepcopy
from functools import partial
from typing import Dict, List, Optional

import gym
import numpy as np
import torch as th
import torch.distributions as D
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc.spaces import th_flatten

log = logging.getLogger(__name__)


class TransformedDistributionWithMean(D.TransformedDistribution):
    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TransformDistribution(nn.Module):
    def __init__(self, transforms: List[D.Transform]):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        return TransformedDistributionWithMean(x, self.transforms)


class GaussianFromEmbedding(nn.Module):
    '''
    Computes a gaussian distribution from a vector input, using separate fully
    connnected layers for both mean and std (i.e. std is learned, too).
    '''

    def __init__(
        self,
        n_in: int,
        n_out: int,
        log_std_min: float = -2,
        log_std_max: float = 20,
    ):
        super().__init__()
        self.mu = nn.Linear(n_in, n_out)
        self.log_std = nn.Linear(n_in, n_out)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, input):
        mu = self.mu(input)
        log_std = self.log_std(input).clamp(self.log_std_min, self.log_std_max)
        return D.Normal(mu, log_std.exp())


class GaussianFromMuLogStd(nn.Module):
    def __init__(self, log_std_min: float = -2, log_std_max: float = 20):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, input):
        mu, log_std = input.chunk(2, dim=-1)
        return D.Normal(
            mu, log_std.clamp(self.log_std_min, self.log_std_max).exp()
        )


class CategoricalFromEmbedding(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.logits = nn.Linear(n_in, n_out)

    def forward(self, input):
        logits = self.logits(input)
        return D.Categorical(logits=logits)


class MHGaussianFromEmbedding(nn.Module):
    '''
    Computes N gaussian distributions from a vector input, using separate fully
    connnected layers for both mean and std (i.e. std is learned, too).
    '''

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_heads: int,
        log_std_min: float = -2,
        log_std_max: float = 20,
    ):
        super().__init__()
        self.mu = nn.Linear(n_in, n_out * n_heads)
        self.log_std = nn.Linear(n_in, n_out * n_heads)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.n_heads = n_heads
        self.n_out = n_out

    def forward(self, input):
        mu = self.mu(input)
        log_std = self.log_std(input).clamp(self.log_std_min, self.log_std_max)
        mu = mu.view(-1, self.n_heads, self.n_out)
        log_std = log_std.view(-1, self.n_heads, self.n_out)
        return D.Normal(mu, log_std.exp())


class EmbedDiscreteSkill(nn.Module):
    '''
    A simple embedding layer to map discrete to continuous skills.
    Applies an extra linear layer to the last n_skill elements of the input, and
    replaces these elements with the resulting embedding.
    '''

    def __init__(self, n_skills: int, n_embed: int, activation: str = 'none'):
        super().__init__()
        self.embed = nn.Linear(n_skills, n_embed, bias=False)
        self.n_skills = n_skills
        self.activation = activation

    def forward(self, x):
        o = x.narrow(-1, 0, x.shape[-1] - self.n_skills)
        z = x.narrow(-1, -self.n_skills, self.n_skills)
        e = self.embed(z)
        if self.activation == 'tanh':
            e = th.tanh(e)
        return th.cat([o, e], dim=-1)


class SplitTaskInput(nn.Module):
    '''
    Provides another module with everything else and the task input
    '''

    def __init__(self, space: gym.Space, module: nn.Module):
        super().__init__()
        assert isinstance(
            space, gym.spaces.Dict
        ), f'SplitTaskInput requires a Dict observation space (but got {type(space)})'
        assert (
            'task' in space.spaces
        ), 'SplitTaskInput requires a "task" entry in its input space'
        self._rest_keys = [k for k in space.spaces.keys() if k != 'task']
        self.module = module

    def forward(self, x):
        task_in = x['task']
        rest_in = th.cat(
            [x[key].view(x[key].shape[0], -1) for key in self._rest_keys], dim=1
        )
        return self.module(rest_in, task_in)


class EinsumBilinear(nn.Bilinear):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in1_features, in2_features, out_features, bias)

    def forward(self, x1, x2):
        return th.einsum('bj,ijk,bk->bi', [x1, self.weight, x2]) + self.bias


# From D2RL: Deep Dense Architectures in Reinforcement Learning
class SkipNetwork(nn.Module):
    def __init__(self, n_in: int, n_layers: int, n_hid: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_hid))
        n_hin = n_in + n_hid
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hin, n_hid))

    def forward(self, x):
        y = self.layers[0](x)
        y = F.relu(y, inplace=True)
        for l in self.layers[1:]:
            y = l(th.cat([y, x], dim=1))
            y = F.relu(y, inplace=True)
        return y


class GroupedLinear(nn.Module):
    def __init__(self, inp, outp, groups, bias=True):
        super().__init__()
        self.weight = nn.Parameter(th.Tensor(outp, inp))
        if bias:
            self.bias = nn.Parameter(th.Tensor(outp))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.register_buffer('wmask', th.zeros_like(self.weight))
        assert inp % groups == 0
        assert outp % groups == 0
        c_in = inp // groups
        c_out = outp // groups
        for g in range(groups):
            self.wmask[
                c_out * g : c_out * (g + 1), c_in * g : c_in * (g + 1)
            ].fill_(1)

        self.inp = inp
        self.outp = outp
        self.groups = groups

    def reset_parameters(self) -> None:
        from torch.nn import init

        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight * self.wmask, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.inp, self.outp, self.groups, self.bias is not None
        )


class SkipDoubleQNetwork(nn.Module):
    '''
    Model two Q-function networks inside a single one.
    '''

    def __init__(self, n_in: int, n_layers: int, n_hid: int, n_out: int = 1):
        super().__init__()
        layers = nn.ModuleList()
        self.l0 = nn.Linear(n_in, n_hid * 2)
        n_hin = n_in + n_hid
        for _ in range(n_layers - 1):
            layers.append(GroupedLinear(n_hin * 2, n_hid * 2, 2))
        self.layers = layers
        self.lN = GroupedLinear(n_hid * 2, n_out * 2, 2)
        th.fill_(self.lN.weight.detach(), 0)
        th.fill_(self.lN.bias.detach(), 0)

    def forward(self, x):
        y = self.l0(x)
        y = F.relu(y, inplace=True)
        for i, l in enumerate(self.layers):
            ys = y.chunk(2, dim=1)
            y = l(th.cat([ys[0], x, ys[1], x], dim=1))
            y = F.relu(y, inplace=True)
        y = self.lN(y)
        return y


class ExpandTaskSubgoal(nn.Module):
    def __init__(self, space: gym.Space):
        super().__init__()
        n_task = space.spaces['task'].n
        n_subgoal = space.spaces['subgoal'].shape[0]
        self.output_space = deepcopy(space)
        del self.output_space.spaces['subgoal']
        self.output_space.spaces['subgoal'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_task * n_subgoal,),
            dtype=np.float32,
        )

    def forward(self, x):
        task = x['task']
        subgoal = x['subgoal']
        ts = th.bmm(task.unsqueeze(2), subgoal.unsqueeze(1)).view(
            task.shape[0], -1
        )
        y = copy(x)
        y['subgoal'] = ts
        return y


class BilinearSkipNetwork(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_cond: int,
        n_out: int,
        n_layers: int,
        n_hid: int,
        norm_cond: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EinsumBilinear(n_state, n_cond, n_hid))
        n_hin = n_state + n_hid
        for _ in range(n_layers - 1):
            self.layers.append(EinsumBilinear(n_hin, n_cond, n_hid))
        self.layers.append(EinsumBilinear(n_hid, n_cond, n_out))
        self.norm_cond = norm_cond

    def forward(self, x, c):
        if self.norm_cond > 0:
            c = c / c.norm(dim=1, p=self.norm_cond, keepdim=True)
        y = self.layers[0](x, c)
        y = F.relu(y, inplace=True)
        for l in self.layers[1:-1]:
            y = l(th.cat([y, x], dim=1), c)
            y = F.relu(y, inplace=True)
        return self.layers[-1](y, c)


class FlattenSpace(nn.Module):
    def __init__(self, space: gym.Space):
        super().__init__()
        self._space = space

    def forward(self, d):
        return th_flatten(self._space, d)


class Parallel(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        ys = [m(x) for m in self.mods]
        if isinstance(ys[0], D.Normal):
            loc = th.stack([y.loc for y in ys], dim=1)
            scale = th.stack([y.scale for y in ys], dim=1)
            return D.Normal(loc, scale)
        else:
            return th.stack(ys, dim=1)


class SqueezeLastDim(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


def shorthand(name, *args, **kwargs):
    def inner(fn):
        if name in FactoryType._shorthands:
            raise ValueError(f'Duplicate shorthand: {name}')
        FactoryType._shorthands[name] = partial(fn, *args, **kwargs)
        return fn

    return inner


class FactoryType(type):
    _shorthands = {}

    def __getattr__(self, name):
        if name in self._shorthands:
            return self._shorthands[name]
        raise AttributeError(
            f'type object "{self.__name__}" has no attribute "{name}"'
        )


class Factory(metaclass=FactoryType):
    @staticmethod
    def make(name: str, *args, **kwargs):
        return getattr(Factory, name)(*args, **kwargs)

    @staticmethod
    def double_q(
        obs_space: gym.Space, action_space: gym.Space, q_name: str, **kwargs
    ):
        return Parallel(
            Factory.make(q_name, obs_space, action_space, **kwargs),
            Factory.make(q_name, obs_space, action_space, **kwargs),
        )

    @staticmethod
    @shorthand('q_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('q_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('q_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('q_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('q_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('q_d2rl_d_1024', n_hid=1024, flatten_input=True)
    def q_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space) + gym.spaces.flatdim(action_space)
        if flatten_input:
            if isinstance(action_space, gym.spaces.Dict):
                joint_space = gym.spaces.Dict(
                    dict(**action_space.spaces, **obs_space.spaces)
                )
            else:
                joint_space = gym.spaces.Dict(
                    dict(action=action_space, **obs_space.spaces)
                )
            ms.append(FlattenSpace(joint_space))
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(nn.Linear(n_hid, 1))
        th.nn.init.uniform_(ms[-1].weight, a=-0.003, b=0.003)
        ms.append(SqueezeLastDim())
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('qd_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('qd_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('qd_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('qd_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('qd_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('qd_d2rl_d_1024', n_hid=1024, flatten_input=True)
    def qd_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space) + gym.spaces.flatdim(action_space)
        if flatten_input:
            if isinstance(action_space, gym.spaces.Dict):
                joint_space = gym.spaces.Dict(
                    dict(**action_space.spaces, **obs_space.spaces)
                )
            else:
                joint_space = gym.spaces.Dict(
                    dict(action=action_space, **obs_space.spaces)
                )
            ms.append(FlattenSpace(joint_space))
        ms.append(SkipDoubleQNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('qd_d2rl_flattsg_d_256', n_hid=256)
    @shorthand('qd_d2rl_flattsg_d_512', n_hid=512)
    @shorthand('qd_d2rl_flattsg_d_1024', n_hid=1024)
    def qd_d2rl_flattsg(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        if isinstance(action_space, gym.spaces.Dict):
            joint_space = gym.spaces.Dict(
                dict(**action_space.spaces, **obs_space.spaces)
            )
        else:
            joint_space = gym.spaces.Dict(
                dict(action=action_space, **obs_space.spaces)
            )
        ms.append(ExpandTaskSubgoal(joint_space))
        n_in = gym.spaces.flatdim(ms[0].output_space)
        ms.append(FlattenSpace(ms[0].output_space))
        ms.append(SkipDoubleQNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('qd_relu_d_3x256', n_layers=3, n_hid=256, flatten_input=True)
    def qd_relu(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_layers: int,
        n_hid: int,
        flatten_input: bool,
        activation: str = 'none',
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space) + gym.spaces.flatdim(action_space)
        if flatten_input:
            if isinstance(action_space, gym.spaces.Dict):
                joint_space = gym.spaces.Dict(
                    dict(**action_space.spaces, **obs_space.spaces)
                )
            else:
                joint_space = gym.spaces.Dict(
                    dict(action=action_space, **obs_space.spaces)
                )
            ms.append(FlattenSpace(joint_space))
        ms.append(nn.Linear(n_in, n_hid * 2))
        ms.append(nn.ReLU(inplace=True))
        for i in range(n_layers - 1):
            ms.append(GroupedLinear(n_hid * 2, n_hid * 2, 2))
            ms.append(nn.ReLU(inplace=True))
        ms.append(GroupedLinear(n_hid * 2, 2, 2))
        th.nn.init.uniform_(ms[-1].weight, a=-0.003, b=0.003)
        if activation == 'softplus':
            ms[-1].bias.detach().add_(-3.0)
            ms.append(nn.Softplus())
        elif activation == 'sigmoid':
            ms[-1].bias.detach().add_(-3.0)
            ms.append(nn.Sigmoid())
        elif activation.startswith('sigmoid'):
            ms[-1].bias.detach().add_(-3.0)
            ms.append(nn.Sigmoid())
            ms.append(MulConstant(float(activation[len('sigmoid') :])))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('pi_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('pi_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('pi_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('pi_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('pi_d2rl_d_1024', n_hid=1024, flatten_input=True)
    def pi_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space)
        n_out = gym.spaces.flatdim(action_space)
        if flatten_input:
            ms.append(FlattenSpace(obs_space))
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(
            GaussianFromEmbedding(
                n_in=n_hid, n_out=n_out, log_std_min=-5, log_std_max=2
            )
        )
        ms.append(TransformDistribution([D.TanhTransform(cache_size=1)]))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_d2rl_mt_d_256', n_hid=256, flatten_input=True)
    @shorthand('pi_d2rl_mt_d_512', n_hid=512, flatten_input=True)
    def pi_d2rl_mt(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        spaces = copy(obs_space.spaces)
        del spaces['task']
        mobs_space = gym.spaces.Dict(spaces)
        n_in = gym.spaces.flatdim(mobs_space)
        n_out = gym.spaces.flatdim(action_space)
        if flatten_input:
            ms.append(FlattenSpace(mobs_space))
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(
            MHGaussianFromEmbedding(
                n_in=n_hid,
                n_out=n_out,
                n_heads=obs_space.spaces['task'].n,
                log_std_min=-5,
                log_std_max=2,
            )
        )
        ms.append(TransformDistribution([D.TanhTransform(cache_size=1)]))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_d2rl_discrete_d_256', n_hid=256, flatten_input=True)
    @shorthand('pi_d2rl_discrete_d_512', n_hid=512, flatten_input=True)
    def pi_d2rl_discrete(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space)
        n_out = gym.spaces.flatdim(action_space)
        if flatten_input:
            ms.append(FlattenSpace(obs_space))
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(CategoricalFromEmbedding(n_hid, n_out))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_bilinear_d2rl_d_256', n_hid=256)
    def pi_bilinear_d2rl_d(
        obs_space: gym.Space, action_space: gym.Space, n_hid: int
    ) -> nn.Module:
        assert isinstance(
            obs_space, gym.spaces.Dict
        ), f'pi_bilinear_d2rl_d requires a Dict observation space (but got {type(obs_space)})'
        assert (
            'task' in obs_space.spaces
        ), f'pi_bilinear_d2rl_d requires a "task" observation'
        ms: List[nn.Module] = []
        n_in = int(
            sum(
                [
                    gym.spaces.flatdim(s)
                    for k, s in obs_space.spaces.items()
                    if k != 'task'
                ]
            )
        )
        n_taskin = gym.spaces.flatdim(obs_space.spaces['task'])
        gm = BilinearSkipNetwork(
            n_state=n_in,
            n_cond=n_taskin,
            n_out=gym.spaces.flatdim(action_space) * 2,
            n_layers=4,
            n_hid=n_hid,
            norm_cond=2,
        )
        ms.append(SplitTaskInput(obs_space, gm))
        ms.append(GaussianFromMuLogStd())
        ms.append(TransformDistribution([D.TanhTransform(cache_size=1)]))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('q_bilinear_d2rl_d_256', n_hid=256)
    def q_bilinear_d2rl_d(
        obs_space: gym.Space, action_space: gym.Space, n_hid: int
    ) -> nn.Module:
        assert isinstance(
            obs_space, gym.spaces.Dict
        ), f'q_bilinear_d2rl_d requires a Dict observation space (but got {type(obs_space)})'
        assert (
            'task' in obs_space.spaces
        ), f'q_bilinear_d2rl_d requires a "task" observation'
        ms: List[nn.Module] = []
        n_in = int(
            sum(
                [
                    gym.spaces.flatdim(s)
                    for k, s in obs_space.spaces.items()
                    if k != 'task'
                ]
            )
        ) + gym.spaces.flatdim(action_space)
        n_taskin = gym.spaces.flatdim(obs_space.spaces['task'])
        space = gym.spaces.Dict(dict(action=action_space, **obs_space.spaces))
        gm = BilinearSkipNetwork(
            n_state=n_in,
            n_cond=n_taskin,
            n_out=1,
            n_layers=4,
            n_hid=n_hid,
            norm_cond=2,
        )
        th.fill_(gm.layers[-1].weight.detach(), 0)
        th.fill_(gm.layers[-1].bias.detach(), 0)
        ms.append(SplitTaskInput(space, gm))
        ms.append(SqueezeLastDim())
        return nn.Sequential(*ms)

    @staticmethod
    def pi_diayn_cskill(
        obs_space: gym.Space,
        action_space: gym.Space,
        base: str,
        n_skills: int,
        n_embed: int,
        activation: str = 'none',
    ) -> nn.Module:
        # XXX This is quite hacky as it assumes the [obs, skill] format
        # defined in DIAYNAgent.effective_observation_space().
        actual_obs_dim = gym.spaces.flatdim(obs_space) - n_skills
        pi_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(actual_obs_dim + n_embed,),
            dtype=np.float32,
        )
        base_model = Factory.make(base, pi_obs_space, action_space)
        embed = EmbedDiscreteSkill(n_skills, n_embed, activation)
        assert isinstance(base_model, nn.Sequential)
        return nn.Sequential(embed, base_model)

    @staticmethod
    @shorthand('phi_diayn_d2rl_256', n_hid=256)
    @shorthand('phi_diayn_d2rl_1024', n_hid=1024)
    def phi_diayn_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        n_skills: int,
        obs_dim: int = None,
    ) -> nn.Module:
        # XXX Need to subtract n_skills from the observation space as it already
        # includes the one-hot skill encoding.
        if obs_dim is None:
            obs_dim = gym.spaces.flatdim(obs_space) - n_skills
        return nn.Sequential(
            SkipNetwork(n_in=obs_dim, n_layers=4, n_hid=n_hid),
            nn.Linear(n_hid, n_skills),
        )


def make_model(
    cfg: DictConfig, obs_space: gym.Space, action_space: gym.Space
) -> nn.Module:
    if isinstance(cfg, str):
        return Factory.make(cfg, obs_space, action_space)
    elif isinstance(cfg, DictConfig):
        if 'name' in cfg.keys():
            name = cfg['name']
            args = {k: v for k, v in cfg.items() if k != 'name'}
            return Factory.make(name, obs_space, action_space, **args)
        else:
            models: Dict[str, nn.Module] = {}
            for k, v in cfg.items():
                if v is not None:
                    models[k] = make_model(v, obs_space, action_space)
        return nn.ModuleDict(models)
    else:
        raise ValueError(
            f'Can\'t handle model config: {cfg.pretty(resolve=True)}'
        )
