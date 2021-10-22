# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import platform
import random
import sys
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import gym
import numpy as np
import torch as th
from gym.error import AlreadyPendingCallError
from gym.vector import SyncVectorEnv
from gym.vector.utils import concatenate, write_to_shared_memory
from gym.wrappers import FlattenObservation, TimeLimit
from omegaconf import DictConfig

from hucc.envs.thmp_vector_env import AsyncState
from hucc.envs.thmp_vector_env import TorchAsyncVectorEnv as AsyncVectorEnv

log = logging.getLogger(__name__)


class CompatSyncVectorEnv(SyncVectorEnv):
    '''
    SyncVectorEnv with reset_if_done(), and without support for copying
    observations.
    '''

    def __init__(self, env_fns, observation_space=None, action_space=None):
        super().__init__(env_fns, observation_space, action_space, copy=False)
        self._observations = []

    def reset_if_done(self):
        observations = []
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                observations.append(env.reset())
            else:
                observations.append(self._observations[i])
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        return self.observations

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        self._observations = observations
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        return self.observations

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(
                action
            )
            observations.append(observation)
            infos.append(info)
        self._observations = observations
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        return self.observations, self._rewards, self._dones, infos


class VecPyTorch(gym.Wrapper):
    '''
    Wraps a CompatSyncVectorEnv or AsynVectorEnv, ensures inputs and outputs to
    functions are PyTorch tensors, and advertises action and observation spaces
    for a *single* wrapped environment.
    '''

    def __init__(self, venv: gym.Env, device: str):
        if not isinstance(venv, CompatSyncVectorEnv) and not isinstance(
            venv, AsyncVectorEnv
        ):
            raise ValueError(
                f'This wrapper works with CompatSyncVectorEnv and AsynVectorEnv only; got {type(venv)}'
            )
        super().__init__(venv)
        self._device = th.device(device)

        # This is a bit hacky, but we'll add a free-form context dictionary to
        # the environment so that agents can store some state in there.
        self._ctx: Dict[str, Any] = {}

        # VectorEnv sets the action and observation space to a joint space.
        # However, we simply want to assume an extra 'batch' dimension on
        # everything.
        self.action_space = venv.single_action_space
        self.observation_space = venv.single_observation_space

    @property
    def device(self):
        return self._device

    @property
    def ctx(self):
        return self._ctx

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def _from_np(
        self, x: np.ndarray, dtype: th.dtype = th.float32
    ) -> th.Tensor:
        return th.from_numpy(x).to(dtype=dtype, device=self.device, copy=True)

    def seed(self, seeds):
        # The gym vector wrappers interpret a single integer seed as [seed,
        # seed+1, seed+2, ...]. Let's fix this.
        if seeds is None:
            seeds = [None] * self.num_envs
        if isinstance(seeds, int):
            seeds = [seeds] * self.num_envs
        assert len(seeds) == self.num_envs
        return self.env.seed(seeds)

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        return obs

    def reset_if_done(self):
        if isinstance(self.env, CompatSyncVectorEnv):
            obs = self.env.reset_if_done()
        else:
            self.env._assert_is_running()
            if self.env._state != AsyncState.DEFAULT:
                raise AlreadyPendingCallError(
                    'Calling `reset_if_done` while waiting '
                    'for a pending call to `{0}` to complete'.format(
                        self.env._state.value
                    ),
                    self.env._state.value,
                )

            for pipe in self.env.parent_pipes:
                pipe.send(('reset_if_done', None))
            self.env._state = AsyncState.WAITING_RESET

            obs = self.env.reset_wait()

        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        return obs

    def step(self, actions):
        if isinstance(actions, th.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        elif isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()

        obs, reward, done, info = self.env.step(actions)
        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        reward = self._from_np(reward).unsqueeze(dim=1)
        done = self._from_np(done, dtype=th.bool).unsqueeze(dim=1)
        return obs, reward, done, info

    def render_single(self, index: int = 0, **kwargs):
        if isinstance(self.env, CompatSyncVectorEnv):
            # Copy resulting array; it may have negative strides and hence can't
            # be readily converted to a torch tensor
            return th.from_numpy(self.env.envs[index].render(**kwargs).copy())

        self.env.parent_pipes[index].send(('render', kwargs))
        out, success = self.env.parent_pipes[index].recv()
        # TODO propagate exception if any
        return th.from_numpy(out)

    def render_all(self, **kwargs):
        if isinstance(self.env, CompatSyncVectorEnv):
            # Copy resulting array; it may have negative strides and hence can't
            # be readily converted to a torch tensor
            return [
                th.from_numpy(e.render(**kwargs).copy()) for e in self.env.envs
            ]

        for pipe in self.env.parent_pipes:
            pipe.send(('render', kwargs))
        outs = []
        for pipe in self.env.parent_pipes:
            out, success = pipe.recv()
            # TODO propagate exception if any
            outs.append(th.from_numpy(out))
        return outs

    def call(self, fn: str, *args, **kwargs):
        if isinstance(self.env, CompatSyncVectorEnv):
            return [getattr(e, fn)(*args, **kwargs) for e in self.env.envs]

        for pipe in self.env.parent_pipes:
            pipe.send(('call', {'fn': fn, 'args': args, 'kwargs': kwargs}))
        outs = []
        for pipe in self.env.parent_pipes:
            out, success = pipe.recv()
            # TODO propagate exception if any
            outs.append(out)
        return outs


def async_worker_shared_memory(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    is_done = True
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset' or command == 'reset_if_done':
                if is_done or command == 'reset':
                    observation = env.reset()
                    write_to_shared_memory(
                        index, observation, shared_memory, observation_space
                    )
                is_done = False
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                is_done = done
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == 'render':
                rendered = env.render(**data)
                pipe.send((rendered, True))
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            elif command == 'call':
                ret = getattr(env, data['fn'])(*data['args'], **data['kwargs'])
                pipe.send((ret, True))
            else:
                raise RuntimeError(
                    'Received unknown command `{0}`. Must '
                    'be one of `reset`, `step`, `seed`, `close`, '
                    '`render`, `_check_observation_space`, `reset_if_done`, `call`.'.format(
                        command
                    )
                )
    except (KeyboardInterrupt, Exception):
        traceback.print_exc()
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class RewardAccWrapper(gym.Wrapper):
    '''
    Acculumates rewards on a per-episode basis.
    '''

    def __init__(self, env):
        super().__init__(env)
        self._acc = 0.0

    def reset(self, **kwargs):
        self._acc = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._acc += reward
        info['reward_acc'] = self._acc
        return observation, reward, done, info


class FrameCounter(gym.ObservationWrapper):
    '''
    Adds a frame counter to the observation.
    The resulting observation space will be a dictionary, with an additional
    ['time'] entry.
    '''

    def __init__(self, env):
        super().__init__(env)
        self._time = np.array([0], dtype=np.int32)

        maxint = np.iinfo(np.int32).max
        if isinstance(env.observation_space, gym.spaces.Dict):
            self._wrap_in_dict = False
            self.observation_space = gym.spaces.Dict(
                dict(
                    time=gym.spaces.Box(
                        low=0, high=maxint, shape=(1,), dtype=np.int32
                    ),
                    **env.observation_space.spaces,
                )
            )
        else:
            self._wrap_in_dict = True
            self.observation_space = gym.spaces.Dict(
                dict(
                    time=gym.spaces.Box(
                        low=0, high=maxint, shape=(1,), dtype=np.int32
                    ),
                    observation=env.observation_space,
                )
            )

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._time[0] = 0
        return self.observation(observation)

    def step(self, action):
        self._time[0] += 1
        return super().step(action)

    def observation(self, observation):
        if self._wrap_in_dict:
            observation = {'observation': observation}
        observation['time'] = self._time
        return observation


class DictObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            observation=env.observation_space
        )

    def observation(self, observation):
        return {'observation': observation}


class BiskFeatures(gym.ObservationWrapper):
    def __init__(self, env, features: str):
        from bisk import BiskSingleRobotEnv

        super().__init__(env)
        assert isinstance(
            env.unwrapped, BiskSingleRobotEnv
        ), 'BiskFeatures requires a BiskSingleRobotEnv environment'
        assert isinstance(env.observation_space, gym.spaces.Dict)
        if ':' in features:
            dest, features = features.split(':')
        else:
            dest, features = features, features
        self.featurizer = env.unwrapped.make_featurizer(features)
        self.dest = dest
        d = {self.dest: self.featurizer.observation_space}
        for k, v in env.observation_space.spaces.items():
            d[k] = v
        self.observation_space = gym.spaces.Dict(d)

    def observation(self, observation):
        observation[self.dest] = self.featurizer()
        return observation


def make_vec_envs(
    env_name: str,
    n: int,
    device: str = 'cpu',
    seed: Optional[int] = None,
    fork: Optional[bool] = None,
    wrappers: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
    **env_args,
) -> VecPyTorch:
    def make_env(seed, fork, i):
        def thunk():
            env = gym.make(env_name, **env_args)
            if fork and seed is not None:
                random.seed(seed + i)
            if seed is not None and hasattr(env, 'seed'):
                env.seed(seed + i)
            if wrappers:
                for w in wrappers:
                    env = w(env)
            env = RewardAccWrapper(env)
            return env

        return thunk

    fork = n > 1 if fork is None else fork
    if platform.system() == 'Darwin':
        log.info('Disabling forking on macOS due to poor support')
        fork = False

    envs = [make_env(seed, fork, i) for i in range(n)]
    if fork:
        envs = AsyncVectorEnv(
            envs,
            shared_memory=True,
            worker=async_worker_shared_memory,
            copy=False,
        )
    else:
        envs = CompatSyncVectorEnv(envs)
    envs = VecPyTorch(envs, device)
    if seed is not None:
        envs.action_space.seed(seed)

    return envs


def make_wrappers(cfg: DictConfig) -> List:
    wrappers = []
    wrapper_map = {
        'dict_obs': lambda env: DictObs(env),
        'flatten_obs': lambda env: FlattenObservation(env),
        'frame_counter': lambda env: FrameCounter(env),
        'time': lambda env: FrameCounter(env),
    }
    wrapper_map_arg1 = {
        'time_limit': lambda arg: lambda env: TimeLimit(
            env, max_episode_steps=int(arg)
        ),
        'bisk_features': lambda arg: lambda env: BiskFeatures(env, arg),
    }
    for w in cfg.wrappers:
        if isinstance(w, DictConfig):
            if len(w) > 1:
                raise ValueError(f'Malformed wrapper item: {w}')
            for k, arg in w.items():
                if not k in wrapper_map_arg1:
                    raise ValueError(f'No such wrapper: {k}')
                wrappers.append(wrapper_map_arg1[k](arg))
                break
        else:
            if not w in wrapper_map:
                raise ValueError(f'No such wrapper: {w}')
            wrappers.append(wrapper_map[w])
    return wrappers
