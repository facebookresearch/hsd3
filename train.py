# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import json
import logging
import os
import shutil
from copy import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

import hucc
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.hashcount import HashingCountReward
from hucc.spaces import th_flatten

log = logging.getLogger(__name__)


class TrainingSetup(SimpleNamespace):
    cfg: DictConfig
    agent: hucc.Agent
    model: nn.Module
    tbw: SummaryWriter
    viz: Visdom
    rq: hucc.RenderQueue
    envs: hucc.VecPyTorch
    eval_envs: hucc.VecPyTorch
    eval_fn: Callable  # Callable[[TrainingSetup, int], None]
    n_samples: int = 0
    replaybuffer_checkpoint_path: str = 'replaybuffer.pt'
    training_state_path: str = 'training_state.json'
    hcr_checkpoint_path: str = 'hashcounts.pt'
    hcr_space = None
    hcr: HashingCountReward = None

    def close(self):
        self.rq.close()
        self.envs.close()
        self.eval_envs.close()

        # The replay buffer checkpoint may be huge and we won't need it anymore
        # after training is done.
        try:
            Path(self.replaybuffer_checkpoint_path).unlink()
        except FileNotFoundError:
            pass


def setup_training(cfg: DictConfig) -> TrainingSetup:
    if cfg.device == 'cuda' and not th.cuda.is_available():
        log.warning('CUDA not available, falling back to CPU')
        cfg.device = 'cpu'
    # TODO doesn't work with submitit?
    # if th.backends.cudnn.is_available():
    #    th.backends.cudnn.benchmark = True

    th.manual_seed(cfg.seed)
    viz = Visdom(
        server=f'http://{cfg.visdom.host}',
        port=cfg.visdom.port,
        env=cfg.visdom.env,
        offline=cfg.visdom.offline,
        log_to_filename=cfg.visdom.logfile,
    )
    rq = hucc.RenderQueue(viz)

    wrappers = hucc.make_wrappers(cfg.env)
    envs = hucc.make_vec_envs(
        cfg.env.name,
        cfg.env.train_procs,
        device=cfg.device,
        seed=cfg.seed,
        wrappers=wrappers,
        **cfg.env.train_args,
    )
    eval_envs = hucc.make_vec_envs(
        cfg.env.name,
        cfg.env.eval_procs,
        device=cfg.device,
        seed=cfg.seed,
        wrappers=wrappers,
        **cfg.env.eval_args,
    )

    observation_space = hucc.effective_observation_space(cfg.agent, envs)
    action_space = hucc.effective_action_space(cfg.agent, envs)

    def make_model_rec(mcfg, obs_space, action_space) -> nn.Module:
        if isinstance(obs_space, dict) and isinstance(action_space, dict):
            assert set(obs_space.keys()) == set(action_space.keys())
            models: Dict[str, nn.Module] = {}
            for k in obs_space.keys():
                models[k] = make_model_rec(
                    mcfg[k], obs_space[k], action_space[k]
                )
            return nn.ModuleDict(models)
        return hucc.make_model(mcfg, obs_space, action_space)

    model = make_model_rec(cfg.model, observation_space, action_space)
    log.info(f'Model from config:\n{model}')
    model.to(cfg.device)
    optim = hucc.make_optim(cfg.optim, model)

    agent = hucc.make_agent(cfg.agent, envs, model, optim)

    # If the current directoy is different from the original one, assume we have
    # a dedicated job directory. We'll just write our summaries to 'tb/' then.
    try:
        if os.getcwd() != hydra.utils.get_original_cwd():
            tbw = SummaryWriter('tb')
        else:
            tbw = SummaryWriter()
        agent.tbw = tbw
    except:
        # XXX hydra.utils.get_original_cwd throws if we don't run this via
        # run_hydra
        tbw = None

    try:
        no_gs_obs = copy(envs.observation_space.spaces)
        for key in [k for k in no_gs_obs.keys() if k.startswith('_')]:
            del no_gs_obs[key]
        if 'time' in no_gs_obs:
            del no_gs_obs['time']
        no_gs_obs = gym.spaces.Dict(no_gs_obs)
    except:
        no_gs_obs = envs.observation_space

    dump_sc = int(cfg.dump_state_counts)
    if dump_sc > 0:
        hcr = HashingCountReward(gym.spaces.flatdim(no_gs_obs)).to_(cfg.device)
    else:
        hcr = None

    return TrainingSetup(
        cfg=cfg,
        agent=agent,
        model=model,
        tbw=tbw,
        viz=viz,
        rq=rq,
        envs=envs,
        eval_envs=eval_envs,
        eval_fn=eval,
        hcr_space=no_gs_obs,
        hcr=hcr,
    )


def eval(setup: TrainingSetup, n_samples: int = -1):
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    envs = setup.eval_envs

    envs.seed(list(range(envs.num_envs)))  # Deterministic evals
    obs = envs.reset()
    reward = th.zeros(envs.num_envs)
    rewards: List[th.Tensor] = []
    dones: List[th.Tensor] = [th.tensor([False] * envs.num_envs)]
    rq_in: List[List[Dict[str, Any]]] = [[] for _ in range(envs.num_envs)]
    n_imgs = 0
    collect_img = cfg.eval.video is not None
    collect_all = collect_img and cfg.eval.video.record_all
    annotate = collect_img and (
        cfg.eval.video.annotations or (cfg.eval.video.annotations is None)
    )
    vwidth = int(cfg.eval.video.size[0]) if collect_img else 0
    vheight = int(cfg.eval.video.size[1]) if collect_img else 0
    metrics = set(cfg.eval.metrics.keys())
    metrics_v: Dict[str, Any] = defaultdict(
        lambda: [[] for _ in range(envs.num_envs)]
    )
    extra = None
    entropy_ds = []
    while True:
        if collect_img:
            extra_right: List[List[str]] = [[] for _ in range(envs.num_envs)]
            if extra is not None and isinstance(extra, dict) and 'viz' in extra:
                for i in range(envs.num_envs):
                    for k in extra['viz']:
                        if isinstance(extra[k][i], str):
                            extra_right[i].append(f'{k} {extra[k][i]}')
                        elif isinstance(extra[k][i], np.ndarray):
                            v = np.array2string(
                                extra[k][i], separator=',', precision=2
                            )
                            extra_right[i].append(f'{k} {v}')
                        else:
                            v = np.array2string(
                                extra[k][i].cpu().numpy(),
                                separator=',',
                                precision=2,
                            )
                            extra_right[i].append(f'{k} {v}')
            if collect_all:
                for i, img in enumerate(
                    envs.render_all(
                        mode='rgb_array', width=vwidth, height=vheight
                    )
                ):
                    if dones[-1][i].item():
                        continue
                    rq_in[i].append(
                        {
                            'img': img,
                            's_left': [
                                f'Eval',
                                f'Samples {n_samples}',
                            ],
                            's_right': [
                                f'Trial {i+1}',
                                f'Frame {len(rewards)}',
                                f'Reward {reward[i].item():+.02f}',
                            ]
                            + extra_right[i],
                        }
                    )
            else:
                if not dones[-1][0].item():
                    rq_in[0].append(
                        {
                            'img': envs.render_single(
                                mode='rgb_array', width=vwidth, height=vheight
                            ),
                            's_left': [
                                f'Eval',
                                f'Samples {n_samples}',
                            ],
                            's_right': [
                                f'Frame {n_imgs}',
                                f'Reward {reward[0].item():+.02f}',
                            ]
                            + extra_right[0],
                        }
                    )
                    n_imgs += 1
                    if n_imgs > cfg.eval.video.length:
                        collect_img = False

        action, extra = agent.action(envs, obs)
        next_obs, reward, done, info = envs.step(action)
        if 'entropy_d' in envs.ctx:
            entropy_ds.append(envs.ctx['entropy_d'])

        for k in metrics:
            for i in range(len(info)):
                if dones[-1][i].item():
                    continue
                if k in info[i]:
                    metrics_v[k][i].append(info[i][k])
        rewards.append(reward.view(-1).to('cpu', copy=True))
        dones.append(done.view(-1).cpu() | dones[-1])
        if dones[-1].all():
            break
        obs = envs.reset_if_done()

    reward = th.stack(rewards, dim=1)
    not_done = th.logical_not(th.stack(dones, dim=1))
    r_undiscounted = (reward * not_done[:, :-1]).sum(dim=1)
    r_discounted = reward.clone()
    discounted_bwd_cumsum_(r_discounted, cfg.agent.gamma, mask=not_done[:, 1:])[
        :, 0
    ]
    ep_len = not_done.to(th.float32).sum(dim=1)

    metrics_v['episode_length'] = ep_len
    metrics_v['return_disc'] = r_discounted
    metrics_v['return_undisc'] = r_undiscounted
    default_agg = ['mean', 'min', 'max', 'std']
    for k, v in metrics_v.items():
        agg = cfg.eval.metrics[k]
        if isinstance(agg, str):
            if ':' in agg:
                epagg, tagg = agg.split(':')
                if epagg == 'final':
                    v = [ev[-1] for ev in v]
                elif epagg == 'max':
                    v = [max(ev) for ev in v]
                elif epagg == 'min':
                    v = [min(ev) for ev in v]
                elif epagg == 'sum':
                    v = [sum(ev) for ev in v]
                agg = tagg
            elif not isinstance(v, th.Tensor):
                v = itertools.chain(v)
            if agg == 'default':
                agg = default_agg
            else:
                agg = [agg]
        if isinstance(v, th.Tensor):
            agent.tbw_add_scalars(f'Eval/{k}', v, agg, n_samples)
        else:
            agent.tbw_add_scalars(
                f'Eval/{k}', th.tensor(v).float(), agg, n_samples
            )
    log.info(
        f'eval done, avg len {ep_len.mean().item():.01f}, avg return {r_discounted.mean().item():+.03f}, undisc avg {r_undiscounted.mean():+.03f} min {r_undiscounted.min():+0.3f} max {r_undiscounted.max():+0.3f}'
    )

    if len(entropy_ds) > 0:
        ent_d = (
            th.stack(entropy_ds)
            .T.to(not_done.device)
            .masked_select(not_done[:, :-1])
        )
        agent.tbw_add_scalar('Eval/EntropyDMean', ent_d.mean(), n_samples)
        agent.tbw.add_histogram('Eval/EntropyD', ent_d, n_samples, bins=20)

    if sum([len(q) for q in rq_in]) > 0:
        # Display cumulative reward in video
        c_rew = reward * not_done[:, :-1]
        for i in range(c_rew.shape[1] - 1):
            c_rew[:, i + 1] += c_rew[:, i]
        n_imgs = 0
        for i, ep in enumerate(rq_in):
            for j, input in enumerate(ep):
                if n_imgs <= cfg.eval.video.length:
                    input['s_right'].append(f'Acc. Reward {c_rew[i][j]:+.02f}')
                    if annotate:
                        rq.push(**input)
                    else:
                        rq.push(img=input['img'])
                    n_imgs += 1
        rq.plot()


def train_loop(setup: TrainingSetup):
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    envs = setup.envs

    agent.train()

    n_envs = envs.num_envs
    cp_path = cfg.checkpoint_path
    record_videos = cfg.video is not None
    annotate = record_videos and (
        cfg.video.annotations or (cfg.video.annotations is None)
    )
    vwidth = int(cfg.video.size[0]) if record_videos else 0
    vheight = int(cfg.video.size[1]) if record_videos else 0
    max_steps = int(cfg.max_steps)
    dump_sc = int(cfg.dump_state_counts)
    obs = envs.reset()
    n_imgs = 0
    collect_img = False
    agent.train()
    while setup.n_samples < max_steps:
        if setup.n_samples % cfg.eval.interval == 0:
            # Checkpoint time
            try:
                log.debug(
                    f'Checkpointing to {cp_path} after {setup.n_samples} samples'
                )
                with open(cp_path, 'wb') as f:
                    agent.save_checkpoint(f)
                if cfg.keep_all_checkpoints:
                    p = Path(cp_path)
                    cp_unique_path = str(
                        p.with_name(
                            p.stem + f'_{setup.n_samples:08d}' + p.suffix
                        )
                    )
                    shutil.copy(cp_path, cp_unique_path)
            except:
                log.exception('Checkpoint saving failed')

            agent.eval()
            setup.eval_fn(setup, setup.n_samples)
            agent.train()

        if record_videos and setup.n_samples % cfg.video.interval == 0:
            collect_img = True
            pass
        if collect_img:
            rqin = {
                'img': envs.render_single(
                    mode='rgb_array', width=vwidth, height=vheight
                )
            }
            if annotate:
                rqin['s_left'] = [
                    f'Samples {setup.n_samples}',
                    f'Frame {n_imgs}',
                ]
                rqin['s_right'] = [
                    'Train',
                ]
            rq.push(**rqin)
            n_imgs += 1
            if n_imgs > cfg.video.length:
                rq.plot()
                n_imgs = 0
                collect_img = False
        action, extra = agent.action(envs, obs)
        next_obs, reward, done, info = envs.step(action)
        agent.step(envs, obs, action, extra, (next_obs, reward, done, info))
        if dump_sc > 0:
            if setup.n_samples % dump_sc == 0:
                d = len(setup.hcr.bucket_sizes)
                sc = setup.hcr.tables.clamp(max=1).sum().item() / d
                agent.tbw_add_scalar(f'Train/UniqueStates', sc, setup.n_samples)
            setup.hcr.inc_hash(th_flatten(setup.hcr_space, obs))
        obs = envs.reset_if_done()
        setup.n_samples += n_envs

    # Final checkpoint & eval time
    try:
        log.debug(f'Checkpointing to {cp_path} after {setup.n_samples} samples')
        with open(cp_path, 'wb') as f:
            agent.save_checkpoint(f)
        if cfg.keep_all_checkpoints:
            p = Path(cp_path)
            cp_unique_path = str(
                p.with_name(p.stem + f'_{setup.n_samples:08d}' + p.suffix)
            )
            shutil.copy(cp_path, cp_unique_path)
    except:
        log.exception('Checkpoint saving failed')

    agent.eval()
    setup.eval_fn(setup, setup.n_samples)
    agent.train()


def checkpoint(setup):
    log.info('Checkpointing agent and replay buffer')
    cfg = setup.cfg
    cp_path = cfg.checkpoint_path
    try:
        with open(cp_path, 'wb') as f:
            setup.agent.save_checkpoint(f)
    except:
        log.exception('Checkpointing agent failed')

    if hasattr(setup.agent, '_buffer'):
        try:
            with open(setup.replaybuffer_checkpoint_path, 'wb') as f:
                setup.agent._buffer.save(f)
        except:
            log.exception('Checkpointing replay buffer failed')

    if setup.hcr is not None:
        try:
            with open(setup.hcr_checkpoint_path, 'wb') as f:
                setup.hcr.save(f)
        except:
            log.exception('Checkpointing hashcounts failed')

    try:
        with open(setup.training_state_path, 'wt') as f:
            json.dump(
                {
                    'n_samples': setup.n_samples,
                },
                f,
            )
    except:
        log.exception('Checkpointing training state failed')


def restore(setup):
    ts_path = setup.training_state_path
    if Path(ts_path).is_file():
        try:
            with open(ts_path, 'rt') as f:
                d = json.load(f)
            setup.n_samples = d['n_samples']
        except:
            log.exception('Restoring training state failed')
    else:
        return

    cfg = setup.cfg
    cp_path = cfg.checkpoint_path
    if cp_path and Path(cp_path).is_file():
        log.info(f'Loading agent from checkpoint {cp_path}')
        with open(cp_path, 'rb') as fd:
            setup.agent.load_checkpoint(fd)
    else:
        raise RuntimeError('Found training state but no agent checkpoint')

    rpbuf_path = setup.replaybuffer_checkpoint_path
    if hasattr(setup.agent, '_buffer') and Path(rpbuf_path).is_file():
        try:
            with open(rpbuf_path, 'rb') as f:
                setup.agent._buffer.load(f)
        except:
            log.exception('Restoring replay buffer failed')

    hcr_path = setup.hcr_checkpoint_path
    if hasattr(setup, 'hcr') and Path(hcr_path).is_file():
        try:
            with open(setup.hcr_checkpoint_path, 'rb') as f:
                setup.hcr.load(f)
        except:
            log.exception('Restoring hashcounts failed')


def auto_adapt_config(cfg: DictConfig) -> DictConfig:
    if cfg.env.name.startswith('BiskStairs'):
        # Goal space should be postfixed with "-relz" since Z features reported
        # by this environment are wrt to the current geom under the robot
        if 'goal_space' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.goal_space = f'{cfg.goal_space}-relz'
            OmegaConf.set_struct(cfg, True)
    elif cfg.env.name.startswith('BiskPoleBalance'):
        # High-level acting at every time-step
        if 'action_interval' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.action_interval = 1
            OmegaConf.set_struct(cfg, True)
    return cfg


@hydra.main(config_path='config')
def main(cfg: DictConfig):
    log.info(f'** running from source tree at {hydra.utils.get_original_cwd()}')
    if cfg.auto_adapt:
        cfg = auto_adapt_config(cfg)
    log.info(
        f'** configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}'
    )

    setup = setup_training(cfg)
    hucc.set_checkpoint_fn(checkpoint, setup)
    restore(setup)

    train_loop(setup)
    setup.close()


if __name__ == '__main__':
    main()
