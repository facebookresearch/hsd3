# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from types import SimpleNamespace

import gym
from omegaconf import DictConfig
from torch import nn

from hucc.agents.agent import Agent
from hucc.agents.diayn import DIAYNAgent
from hucc.agents.hiro import HIROAgent
from hucc.agents.hsd3 import HSD3Agent
from hucc.agents.hsdb import HSDBAgent
from hucc.agents.sac import SACAgent
from hucc.agents.sachrl import SACHRLAgent
from hucc.agents.sacmt import SACMTAgent
from hucc.agents.sacse import SACSEAgent


def agent_cls(name: str):
    return {
        'diayn': DIAYNAgent,
        'hiro': HIROAgent,
        'hsd3': HSD3Agent,
        'hsdb': HSDBAgent,
        'sac': SACAgent,
        'sachrl': SACHRLAgent,
        'sacmt': SACMTAgent,
        'sacse': SACSEAgent,
    }[name]


def effective_observation_space(cfg: DictConfig, env: gym.Env):
    return agent_cls(cfg.name).effective_observation_space(env=env, cfg=cfg)


def effective_action_space(cfg: DictConfig, env: gym.Env):
    return agent_cls(cfg.name).effective_action_space(env=env, cfg=cfg)


def make_agent(
    cfg: DictConfig, env: gym.Env, model: nn.Module, optim: SimpleNamespace
) -> Agent:
    return agent_cls(cfg.name)(env=env, model=model, optim=optim, cfg=cfg)
