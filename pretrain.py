# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import getpass
import importlib
import json
import logging
import os
import shutil
import uuid
from collections import defaultdict
from copy import copy, deepcopy
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, cast

import gym
import hydra
import numpy as np
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing as mp

import hucc
from hucc.agents.sacmt import SACMTAgent
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.envs.ctrlgs import CtrlgsPreTrainingEnv
from hucc.envs.goal_spaces import g_goal_spaces
from hucc.spaces import th_flatten
from train import TrainingSetup, checkpoint, restore, setup_training

log = logging.getLogger(__name__)


def abstr_name(cfg, d):
    feats = map(int, d.replace('+', ',').split(','))
    if cfg.env.name == 'ContCtrlgsPreTraining-v1':
        return ','.join(
            [
                CtrlgsPreTrainingEnv.feature_name(cfg.robot, cfg.features, f)
                for f in feats
            ]
        )
    return d


def estimate_ctrlb(setup: TrainingSetup) -> Dict[str, Dict[str, float]]:
    agent = cast(SACMTAgent, setup.agent)
    model = setup.model
    cfg = setup.cfg

    buffer = agent._buffer
    if buffer.size < agent._warmup_samples or buffer._b is None:
        k = setup.goal_dims.keys()
        return {'q': {d: 0.0 for d in k}, 'r': {d: 0.0 for d in k}}

    entry_point = gym.envs.registry.spec(cfg.env.name).entry_point
    mod_name, attr_name = entry_point.split(":")
    mod = importlib.import_module(mod_name)
    env_cls = getattr(mod, attr_name)
    gsdim = buffer._b['gs_observation'].shape[1]
    psi, offset = env_cls.abstraction_matrix(cfg.robot, cfg.features, gsdim)
    delta_feats = env_cls.delta_features(cfg.robot, cfg.features)
    psi_1 = np.linalg.inv(psi)
    offset_1 = -np.matmul(offset, psi_1)
    psi = th.tensor(psi, dtype=th.float32, device=cfg.device)
    offset = th.tensor(offset, dtype=th.float32, device=cfg.device)
    psi_1 = th.tensor(psi_1, dtype=th.float32, device=cfg.device)
    offset_1 = th.tensor(offset_1, dtype=th.float32, device=cfg.device)
    task_map = setup.task_map
    task_idx = [0] * len(task_map)
    for k, v in task_map.items():
        task_idx[v] = int(k)
    dscale = agent._gamma ** cfg.horizon
    ctrl_cost = (
        cfg.horizon
        * cfg.env.args.ctrl_cost
        * 0.25
        * setup.envs.action_space.shape[0]
    )

    n = 1024
    cperf: Dict[str, Dict[str, float]] = {'q': {}, 'r': {}}
    starts = th.where(buffer._b['start_state'] == True)[0]
    for d in setup.goal_dims.keys():
        #  Query start states from replay buffer
        idx = th.randint(low=0, high=starts.shape[0], size=(n,))
        obs = buffer._b['obs_observation'][starts[idx]].to(cfg.device)

        # Sample goals and project to input space
        # XXX assumes we train with backprojecting goals
        feats = list(map(int, d.replace('+', ',').split(',')))

        if len(feats) > 1 and cfg.estimate_joint_spaces == 'gmm':
            sidx = th.randint(low=0, high=buffer.size, size=(n * 10,))
            sample = (
                th.bmm(
                    buffer._b['gs_observation'][sidx].unsqueeze(1),
                    psi[feats]
                    .T.unsqueeze(0)
                    .expand(sidx.shape[0], gsdim, len(feats)),
                ).squeeze(1)
                + offset[feats]
            )
            clf = GaussianMixture(
                n_components=32, max_iter=100, n_init=10, covariance_type='full'
            )
            clf.fit(sample.cpu())
            wgoal = th.tensor(
                clf.sample(n)[0].clip(-1, 1),
                device=obs.device,
                dtype=th.float32,
            )
        elif len(feats) > 1 and cfg.estimate_joint_spaces == 'kmeans':
            sidx = th.randint(low=0, high=buffer.size, size=(n * 10,))
            sample = (
                th.bmm(
                    buffer._b['gs_observation'][sidx].unsqueeze(1),
                    psi[feats]
                    .T.unsqueeze(0)
                    .expand(sidx.shape[0], gsdim, len(feats)),
                ).squeeze(1)
                + offset[feats]
            )
            clf = KMeans(n_clusters=n)
            clf.fit(sample.cpu())
            wgoal = th.tensor(
                clf.cluster_centers_.clip(-1, 1),
                device=obs.device,
                dtype=th.float32,
            )
        else:
            wgoal = th.rand(size=(n, len(feats)), device=obs.device) * 2 - 1

        gsobs = buffer._b['gs_observation'][starts[idx]].to(cfg.device)
        ws = (
            th.bmm(
                gsobs.unsqueeze(1),
                psi[feats].T.unsqueeze(0).expand(n, gsdim, len(feats)),
            ).squeeze(1)
            + offset[feats]
        )
        for i, f in enumerate(feats):
            if f in delta_feats:
                wgoal[:, i] += ws[:, i]
        s = gsobs[:, task_idx]
        gb = (
            th.bmm(
                wgoal.unsqueeze(1),
                psi_1[feats].unsqueeze(0).expand(n, len(feats), gsdim),
            ).squeeze(1)
            + offset_1
        )[:, task_idx]
        feature_mask = th.zeros(len(task_idx), device=obs.device)
        for f in d.replace('+', ',').split(','):
            feature_mask[setup.task_map[f]] = 1
        goal = (gb - s) * feature_mask

        # Record distances in goal space
        wobs = (
            th.bmm(
                gsobs.unsqueeze(1),
                psi[feats].T.unsqueeze(0).expand(n, gsdim, len(feats)),
            ).squeeze(1)
            + offset[feats]
        )
        dist = th.linalg.norm(wgoal - wobs, ord=2, dim=1)

        # Finally, the bow task input
        task = th.zeros(len(task_map), device=cfg.device)
        for f in feats:
            task[task_map[str(f)]] = 1
        task = task.unsqueeze(0).expand(n, len(task_map))

        # Query mean action and corresponding Q-value
        with th.no_grad():
            action = model.pi(
                {
                    'observation': obs,
                    'task': task,
                    'desired_goal': goal,
                }
            ).mean
            q = model.q(
                {
                    'observation': obs,
                    'task': task,
                    'desired_goal': goal,
                    'action': action,
                }
            )
            q1 = q[:, 0]
            q2 = q[:, 1]
            r = model.reachability(
                {
                    'observation': obs,
                    'task': task,
                    'desired_goal': goal,
                    'action': action,
                }
            )
        q = th.min(q1, q2).view(-1)
        cperf['q'][d] = (q >= (dist - ctrl_cost) * dscale).sum().item() / n
        cperf['r'][d] = r.clamp(0, 1).mean().item()

    return cperf


def update_fdist(
    setup: TrainingSetup,
    cperf_old: Dict[str, float],
    cperf: Dict[str, float],
    n_samples: int,
):
    cfg = setup.cfg
    agent = setup.agent
    envs = setup.envs
    combine_after_steps = int(cfg.combine_after_steps)

    # Compute learning progress
    lp: Dict[str, float] = {}
    for k in cperf.keys():
        if not k in cperf_old:
            if k in cperf:
                lp[k] = cperf[k]
            else:
                lp[k] = float(cfg.lp_new_task)
        else:
            if k in lp and cfg.task_weighting == 'lp_smooth':
                lp[k] = 0.9 * lp[k] + 0.1 * (cperf[k] - cperf_old[k])
            else:
                lp[k] = cperf[k] - cperf_old[k]
    if 'total' in lp:
        del lp['total']

    # Find abstractions that are already epsilon-controllable, join
    # them to a new one if all possible subsets of the combination are
    # also epsilon-controllable. Examples:
    # - add 3,4 if 3 and 4 are eps-ctrl
    # - add 3,4,5 if 3,4 and 4,5 and 3,5 are eps-ctrl
    eps_ctrl = set()
    for d in setup.goal_dims:
        suffix = ''
        if cperf[d] >= 1.0 - cfg.ctrl_eps:
            eps_ctrl.add(d)
            suffix = '*'
        log.info(
            f'Features {abstr_name(cfg, d)} at ctrl {cperf[d]:.04f}{suffix}'
        )

    # Collect new combinations with a rather optimistic initial
    # performance estimate
    new_abs: Dict[str, float] = {}
    for comb in combinations(eps_ctrl, 2):
        if n_samples < combine_after_steps:
            continue
        feats = sorted(
            set(comb[0].split(',') + comb[1].split(',')),
            key=lambda x: [int(i) for i in x.split('+')],
        )
        if len(feats) > cfg.feature_rank_max:
            continue
        d = ','.join(feats)
        if d in setup.goal_dims:
            continue
        all_eps_ctrl = True
        max_perf = 1.0
        for subset in combinations(feats, len(feats) - 1):
            sd = ','.join(
                sorted(subset, key=lambda x: [int(i) for i in x.split('+')])
            )
            if not sd in eps_ctrl:
                all_eps_ctrl = False
                break
            max_perf *= cperf[sd]
        if not all_eps_ctrl:
            continue
        # So there's our candidate. Assume we get have perfect zero-shot
        # generalization and multiply the controllability of the combined
        # features.
        new_abs[d] = max_perf

    combined = set()
    new_absk = sorted(new_abs.keys(), key=lambda d: -new_abs[d])
    old_weights = copy(setup.goal_dims)
    for d in new_absk[: int(cfg.max_new_tasks)]:
        log.info(f'Adding new abstraction {abstr_name(cfg, d)}')
        setup.goal_dims[d] = 1
        if cfg.task_weighting == 'downrank_new':
            setup.goal_dims[d] = float(cfg.downrank)
        cperf[d] = new_abs[d]
        lp[d] = float(cfg.lp_new_task)
        feats = d.split(',')
        for subset in combinations(feats, len(feats) - 1):
            sd = ','.join(
                sorted(subset, key=lambda x: [int(i) for i in x.split('+')])
            )
            combined.add(sd)

    if cfg.task_weighting == 'lp' or cfg.task_weighting == 'lp_smooth':
        N = len(lp)
        eps = float(cfg.lp_eps)  # 0.4 in CURIOUS
        total_lp = sum((abs(v) for v in lp.values())) + 1e-7
        for k, v in lp.items():
            setup.goal_dims[k] = eps * (1 / N) + (1 - eps) * (abs(v) / total_lp)
    elif cfg.task_weighting == 'downrank_new':
        pass
    elif cfg.task_weighting == 'downrank_combined':
        for k in combined:
            setup.goal_dims[k] = float(cfg.downrank)
    elif cfg.task_weighting == 'tderr':
        assert cfg.agent.name == 'sacmt'
        tderr = cast(SACMTAgent, agent).avg_tderr_per_task()
        max_err = max(tderr.values()) if tderr else 1.0
        for k in setup.goal_dims.keys():
            setup.goal_dims[k] = tderr.get(k, max_err)
    elif cfg.task_weighting == 'uniform':
        N = len(setup.goal_dims)
        for k, v in setup.goal_dims.items():
            setup.goal_dims[k] = 1 / N
    else:
        raise ValueError(
            f'Unknown task weighting: {cfg.task_weighting}; use "lp" or "downrank_combined"'
        )
    for k, v in setup.goal_dims.items():
        delta = v - old_weights.get(k, 0)
        log.debug(
            f'Features {abstr_name(cfg, k)} new weight {v:.04f} ({delta:+.04f})'
        )

    # Set new abstractions for training; for evaluation however we'll
    # keep them around at full probability.
    envs.call('set_goal_dims', setup.goal_dims)
    setup.eval_envs.call(
        'set_goal_dims', {d: 1 for d, _ in setup.goal_dims.items()}
    )
    if agent.tbw:
        agent.tbw.add_scalars('Eval/LearningProgress', lp, agent.n_samples)
        agent.tbw.add_scalars(
            'Eval/NewTaskProbs', setup.goal_dims, agent.n_samples
        )


# This evaluation function returns per-task performances
def eval_mfdim(setup, n_samples: int) -> Dict[str, float]:
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    envs = setup.eval_envs
    n_episodes = cfg.eval.episodes_per_task * len(setup.goal_dims)
    task_map_r: Dict[int, int] = {}
    for k, v in setup.task_map.items():
        task_map_r[v] = int(k)

    envs.seed(list(range(envs.num_envs)))
    obs = envs.reset()
    n_done = 0
    reached_goala: Dict[str, List[bool]] = defaultdict(list)
    reward = th.zeros(envs.num_envs)
    rewards: List[th.Tensor] = []
    dones: List[th.Tensor] = [th.tensor([False] * envs.num_envs)]
    rq_in: List[List[Dict[str, Any]]] = [[] for _ in range(envs.num_envs)]
    n_imgs = 0
    collect_img = cfg.eval.video is not None
    collect_all = collect_img and cfg.eval.video.record_all
    vwidth = int(cfg.eval.video.size[0]) if collect_img else 0
    vheight = int(cfg.eval.video.size[1]) if collect_img else 0

    while True:
        abstractions = []
        for i in range(envs.num_envs):
            bits = list(th.where(obs['task'][i] == 1)[0].cpu().numpy())
            abstractions.append([task_map_r.get(b, b) for b in bits])

        if collect_img:
            if collect_all:
                # TODO This OOMs if we do many evaluations since we record way
                # more frames than we need to.
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
                                f'Features {abstractions[i]}',
                                f'Reward {reward[i].item():+.02f}',
                            ],
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
                                f'Features {abstractions[0]}',
                                f'Reward {reward[0].item():+.02f}',
                            ],
                        }
                    )
                    n_imgs += 1
                    if n_imgs > cfg.eval.video.length:
                        collect_img = False

        t_obs = (
            th_flatten(envs.observation_space, obs)
            if cfg.agent.name != 'sacmt'
            else obs
        )
        action, _ = agent.action(envs, t_obs)
        next_obs, reward, done, info = envs.step(action)

        soft_reset = th.tensor(['SoftReset' in inf for inf in info])
        done = done.view(-1).cpu()
        rewards.append(reward.view(-1).to('cpu', copy=True))
        dones.append(done | soft_reset)

        # Record minimum distance reached for all done environments
        for d in th.where(dones[-1] == True)[0].numpy():
            key = ','.join([str(a) for a in abstractions[d]])
            reached_goala[key].append(info[d]['reached_goal'])

        n_done += dones[-1].sum().item()
        if n_done >= n_episodes:
            break
        obs = envs.reset_if_done()

    reward = th.stack(rewards, dim=1)
    not_done = th.logical_not(th.stack(dones, dim=1))
    r_discounted = reward.clone()
    discounted_bwd_cumsum_(r_discounted, cfg.agent.gamma, mask=not_done[:, 1:])[
        :, 0
    ]
    r_undiscounted = reward.clone()
    discounted_bwd_cumsum_(r_undiscounted, 1.0, mask=not_done[:, 1:])[:, 0]

    # Gather stats regarding which goals were reached
    goals_reached = 0.0
    goalsa_reached: Dict[str, float] = defaultdict(float)
    for abstr, reached in reached_goala.items():
        goalsa_reached[abstr] = th.tensor(reached).sum().item() / len(reached)
        goals_reached += goalsa_reached[abstr] * len(reached)

    goals_reached /= n_done
    goalsa_reached['total'] = goals_reached
    if agent.tbw:
        agent.tbw_add_scalars('Eval/ReturnDisc', r_discounted)
        agent.tbw_add_scalars('Eval/ReturnUndisc', r_undiscounted)
        agent.tbw.add_scalars(
            'Eval/GoalsReached', goalsa_reached, agent.n_samples
        )
        agent.tbw.add_scalars(
            'Eval/NumTrials',
            {a: len(d) for a, d in reached_goala.items()},
            agent.n_samples,
        )
    log.info(
        f'eval done, goals reached {goals_reached:.03f}, avg return {r_discounted.mean().item():+.03f}, undisc avg {r_undiscounted.mean():+.03f} min {r_undiscounted.min():+0.3f} max {r_undiscounted.max():+0.3f}'
    )

    if sum([len(q) for q in rq_in]) > 0:
        # Display cumulative reward in video
        c_rew = reward * not_done[:, :-1]
        for i in range(c_rew.shape[1] - 1):
            c_rew[:, i + 1] += c_rew[:, i]
            c_rew[:, i + 1] *= not_done[:, i]
        n_imgs = 0
        for i, ep in enumerate(rq_in):
            for j, input in enumerate(ep):
                if n_imgs <= cfg.eval.video.length:
                    input['s_right'].append(f'Acc. Reward {c_rew[i][j]:+.02f}')
                    rq.push(**input)
                    n_imgs += 1
        rq.plot()

    return goalsa_reached


def train_loop_mfdim_learner(setup: TrainingSetup, queue: mp.Queue):
    cfg = setup.cfg
    agent = setup.agent
    envs = setup.envs
    n_envs = setup.envs.num_envs
    max_steps = int(cfg.max_steps)

    log.debug(f'learner started')
    agent.train()

    while setup.n_samples < max_steps - n_envs:
        # log.debug(f'learner loop {setup.n_samples} queue size {queue.qsize()}')
        transition = queue.get()
        agent.step(envs, *transition)
        del transition
        setup.n_samples += n_envs


def train_loop_mfdim_actor(setup: TrainingSetup):
    cfg = setup.cfg
    agent = setup.agent
    queues = setup.queues
    rq = setup.rq
    envs = setup.envs
    model = setup.model

    agent.train()

    shared_model = deepcopy(model)
    shared_model.to('cpu')
    # We'll never need gradients for the target network
    for param in shared_model.parameters():
        param.requires_grad_(False)
        param.share_memory_()
    envs.call('set_model', shared_model, agent._gamma)
    prev_n_updates = agent.n_updates

    n_envs = envs.num_envs
    cp_path = cfg.checkpoint_path
    record_videos = cfg.video is not None
    vwidth = int(cfg.video.size[0]) if record_videos else 0
    vheight = int(cfg.video.size[1]) if record_videos else 0
    max_steps = int(cfg.max_steps)
    obs = envs.reset()
    n_imgs = 0
    collect_img = False
    eval_mode = str(cfg.eval_mode)
    cperf: Dict[str, float] = {}
    running_cperf: Dict[str, float] = defaultdict(float)
    while setup.n_samples < max_steps:
        log.debug(f'actor loop {setup.n_samples}')
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
                        p.with_name(f'{p.stem}_{setup.n_samples:08d}{p.suffix}')
                    )
                    shutil.copy(cp_path, cp_unique_path)
            except:
                log.exception('Checkpoint saving failed')

            est = estimate_ctrlb(setup)
            q_cperf = est['q']
            r_cperf = est['r']

            if eval_mode == 'rollouts' or len(running_cperf) == 0:
                agent.eval()
                cperf_new = eval_mfdim(setup, setup.n_samples)
                agent.train()
                if len(running_cperf) == 0:
                    for k, v in cperf_new.items():
                        running_cperf[k] = v
                    del running_cperf['total']
            elif eval_mode == 'running_avg':
                cperf_new = running_cperf
            elif eval_mode == 'q_value':
                cperf_new = q_cperf
            elif eval_mode == 'reachability':
                if not hasattr(model, 'reachability'):
                    log.warning(
                        'Reachability evaluations requested but no reachability model present'
                    )
                cperf_new = r_cperf
            else:
                raise ValueError(f'Unknown evaluation mode {eval_mode}')

            # Fixup goal keys to match '+' syntax
            run_cperf = copy(running_cperf)
            for k in setup.goal_dims.keys():
                flat = k.replace('+', ',')
                if flat == k:
                    continue
                if flat in cperf_new:
                    cperf_new[k] = cperf_new[flat]
                    del cperf_new[flat]
                if flat in run_cperf:
                    run_cperf[k] = run_cperf[flat]
                    del run_cperf[flat]

            if agent.tbw:
                agent.tbw.add_scalars(
                    'Training/GoalsReached', run_cperf, setup.n_samples
                )
                agent.tbw.add_scalars(
                    'Training/CtrlbEstimateQ', q_cperf, setup.n_samples
                )
                agent.tbw.add_scalars(
                    'Training/CtrlbEstimateR', r_cperf, setup.n_samples
                )

            try:
                p = Path(cp_path)
                abs_path = p.with_name(f'{p.stem}_abs.json')
                with open(str(abs_path), 'wt') as ft:
                    json.dump(
                        {
                            'task_map': setup.task_map,
                            'goal_dims': setup.goal_dims,
                            'cperf': cperf_new,
                            'cperf_r': r_cperf,
                            'cperf_q': q_cperf,
                            'cperf_running': run_cperf,
                        },
                        ft,
                    )
                abs_unique_path = p.with_name(
                    f'{p.stem}_{setup.n_samples:08d}_abs.json'
                )
                shutil.copy(str(abs_path), str(abs_unique_path))
            except:
                log.exception('Saving abstraction info failed')

            update_fdist(setup, cperf, cperf_new, setup.n_samples)
            cperf = copy(cperf_new)

        if record_videos and setup.n_samples % cfg.video.interval == 0:
            collect_img = True
        if collect_img:
            rq.push(
                img=envs.render_single(
                    mode='rgb_array', width=vwidth, height=vheight
                ),
                s_left=[
                    f'Samples {setup.n_samples}',
                    f'Frame {n_imgs}',
                ],
                s_right=[
                    'Train',
                ],
            )
            n_imgs += 1
            if n_imgs > cfg.video.length:
                rq.plot()
                n_imgs = 0
                collect_img = False

        t_obs = (
            th_flatten(envs.observation_space, obs)
            if cfg.agent.name != 'sacmt'
            else obs
        )
        action, extra = agent.action(envs, t_obs)
        assert (
            extra is None
        ), "Distributed training doesn't work with extra info from action"
        next_obs, reward, done, info = envs.step(action)
        t_next_obs = (
            th_flatten(envs.observation_space, next_obs)
            if cfg.agent.name != 'sacmt'
            else next_obs
        )
        # XXX CPU transfer seems to be necessary :/
        nq = len(queues)
        ct_obs = {k: v.cpu().chunk(nq) for k, v in t_obs.items()}
        c_action = action.cpu().chunk(nq)
        ct_next_obs = {k: v.cpu().chunk(nq) for k, v in t_next_obs.items()}
        c_done = done.cpu().chunk(nq)
        c_reward = reward.cpu().chunk(nq)
        pos = 0
        for i, queue in enumerate(queues):
            log.debug(
                f'put {c_action[i].shape[0]} of {action.shape[0]} elems into queue {i}'
            )
            n = c_action[i].shape[0]
            queue.put(
                (
                    {k: v[i] for k, v in ct_obs.items()},
                    c_action[i],
                    extra,
                    (
                        {k: v[i] for k, v in ct_next_obs.items()},
                        c_reward[i],
                        c_done[i],
                        info[pos : pos + n],
                    ),
                )
            )
            pos += n
        agent.step(envs, t_obs, action, extra, (t_next_obs, reward, done, info))
        obs = envs.reset_if_done()
        setup.n_samples += n_envs

        # Maintain running average of controllability during training
        for i in range(n_envs):
            if info[i].get('LastStepOfTask', False):
                feats = info[i]['features']
                running_cperf[feats] *= 0.9
                if info[i]['reached_goal']:
                    running_cperf[feats] += 0.1

        # Copy model after update
        if agent.n_updates != prev_n_updates:
            with th.no_grad():
                for tp, dp in zip(
                    shared_model.parameters(), model.parameters()
                ):
                    tp.copy_(dp)
            prev_n_updates = agent.n_updates

    # Final checkpoint & eval time
    try:
        log.debug(f'Checkpointing to {cp_path} after {setup.n_samples} samples')
        with open(cp_path, 'wb') as f:
            agent.save_checkpoint(f)
        if cfg.keep_all_checkpoints:
            p = Path(cp_path)
            cp_unique_path = str(
                p.with_name(f'{p.stem}_{setup.n_samples:08d}{p.suffix}')
            )
            shutil.copy(cp_path, cp_unique_path)
    except:
        log.exception('Checkpoint saving failed')

    agent.eval()
    eval_cperf = eval_mfdim(setup, setup.n_samples)
    agent.train()
    est = estimate_ctrlb(setup)
    q_cperf = est['q']
    r_cperf = est['r']
    if eval_mode == 'rollouts':
        cperf_new = eval_cperf
    elif eval_mode == 'running_avg':
        cperf_new = running_cperf
    elif eval_mode == 'q_value':
        cperf_new = q_cperf
    elif eval_mode == 'reachability':
        if not hasattr(model, 'reachability'):
            log.warning(
                'Reachability evaluations requested but no reachability model present'
            )
        cperf_new = r_cperf
    for d in setup.goal_dims:
        suffix = ''
        if cperf_new[d] >= 1.0 - cfg.ctrl_eps:
            suffix = '*'
        log.info(
            f'Features {abstr_name(cfg, d)} at ctrl {cperf_new[d]:.04f}{suffix}'
        )

    try:
        p = Path(cp_path)
        abs_path = p.with_name(f'{p.stem}_abs.json')
        with open(str(abs_path), 'wt') as ft:
            json.dump(
                {
                    'task_map': setup.task_map,
                    'goal_dims': setup.goal_dims,
                    'cperf': cperf_new,
                    'cperf_eval': eval_cperf,
                    'cperf_r': r_cperf,
                    'cperf_q': q_cperf,
                    'cperf_running': running_cperf,
                },
                ft,
            )
        abs_unique_path = p.with_name(
            f'{p.stem}_{setup.n_samples:08d}_abs.json'
        )
        shutil.copy(str(abs_path), str(abs_unique_path))
    except:
        log.exception('Saving abstraction info failed')


def setup_training_mfdim(cfg: DictConfig):
    if not isinstance(cfg.feature_dims, str):
        cfg.feature_dims = str(cfg.feature_dims)
    gs = g_goal_spaces[cfg.features][cfg.robot]
    n = len(gs['str'])
    # Support some special names for convenience
    if cfg.feature_dims == 'all':
        dims = [str(i) for i in range(n)]
    elif cfg.feature_dims == 'torso':
        dims = [
            str(i)
            for i in range(n)
            if gs['str'][i].startswith(':')
            or gs['str'][i].startswith('torso:')
            or gs['str'][i].startswith('root')
        ]
    else:
        try:
            for d in cfg.feature_dims.split('#'):
                _ = map(int, d.split('+'))
            dims = [d for d in cfg.feature_dims.split('#')]
        except:
            dims = [
                str(i)
                for i in range(n)
                if re.match(cfg.feature_dims, gs['str'][i]) is not None
            ]
    uncontrollable = set()
    for dim in dims:
        for d in map(int, dim.split('+')):
            if not CtrlgsPreTrainingEnv.feature_controllable(
                cfg.robot, cfg.features, d
            ):
                uncontrollable.add(dim)
                log.warning(f'Removing uncontrollable feature {dim}')
                break
    cfg.feature_dims = '#'.join([d for d in dims if not d in uncontrollable])
    if cfg.feature_rank == 'max':
        cfg.feature_rank = len(cfg.feature_dims.split('#'))
    if len(cfg.feature_dims) < int(cfg.feature_rank):
        raise ValueError('Less features to control than the requested rank')

    # Setup custom environment arguments based on the selected robot
    prev_args: Dict[str, Any] = {}
    if isinstance(cfg.env.args, DictConfig):
        prev_args = dict(cfg.env.args)
    cfg.env.args = {
        **prev_args,
        'robot': cfg.robot,
    }
    fdist = {
        ','.join(d): 1.0
        for d in combinations(cfg.feature_dims.split('#'), cfg.feature_rank)
    }
    if cfg.task_weighting.startswith('lp'):
        for k, v in fdist.items():
            fdist[k] = v / len(fdist)
    feats: Set[int] = set()
    task_map: Dict[str, int] = {}
    for fs in fdist.keys():
        for f in map(int, fs.replace('+', ',').split(',')):
            feats.add(f)
    for f in sorted(feats):
        task_map[str(f)] = len(task_map)
    cfg.env.args = {
        **cfg.env.args,
        'feature_dist': fdist,
        'task_map': task_map,
    }
    if cfg.agent.gamma == 'auto_horizon':
        cfg.agent.gamma = 1 - 1 / cfg.horizon
        log.info(f'gamma set to {cfg.agent.gamma}')

    setup = setup_training(cfg)
    if 'goal_dims' in cfg.env.args:
        setup.goal_dims = dict(cfg.env.args.goal_dims)
    else:
        setup.goal_dims = dict(cfg.env.args.feature_dist)
    setup.task_map = dict(cfg.env.args.get('task_map', {}))
    return setup


def worker(rank, role, queues, bcast_barrier, cfg: DictConfig):
    if th.cuda.is_available():
        th.cuda.set_device(rank)
    log.info(
        f'Creating process group of size {cfg.distributed.size} via {cfg.distributed.init_method} [rank={rank}]'
    )
    dist.init_process_group(
        backend='nccl' if th.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=cfg.distributed.size,
        init_method=cfg.distributed.init_method,
    )
    cfg.distributed.role = role

    if role == 'learner':
        OmegaConf.set_struct(cfg.env, False)
        cfg.env.args.fork = False
        cfg.env.eval_procs = 1
        cfg.env.train_procs //= cfg.distributed.num_learners
        cfg.agent.batch_size //= cfg.distributed.num_learners
        cfg.agent.samples_per_update //= cfg.distributed.num_learners
        cfg.agent.warmup_samples //= cfg.distributed.num_learners
    try:
        setup = setup_training_mfdim(cfg)
    except:
        log.exception('Error in training loop')
        raise
    setup.queues = queues
    agent = setup.agent
    agent.bcast_barrier = bcast_barrier

    bcast_barrier.wait()
    if cfg.distributed.num_learners > 1:
        learner_group = dist.new_group(
            [i for i in range(cfg.distributed.num_learners)]
        )
        agent.learner_group = learner_group

    cp_path = cfg.checkpoint_path
    if cfg.init_model_from:
        log.info(f'Initializing model from checkpoint {cfg.init_model_from}')
        with open(cfg.init_model_from, 'rb') as fd:
            data = th.load(fd)
            setup.model.load_state_dict(data['_model'])
            agent._log_alpha.clear()
            for k, v in data['_log_alpha'].items():
                agent._log_alpha[k] = v

    restore(setup)

    log.debug(f'broadcast params {rank}:{role}')
    bcast_barrier.wait()
    for p in setup.model.parameters():
        dist.broadcast(p, src=cfg.distributed.num_learners)
    dist.barrier()
    log.debug('done')

    setup.eval_fn = eval_mfdim
    agent.role = role
    try:
        if role == 'actor':
            hucc.set_checkpoint_fn(checkpoint, setup)
            train_loop_mfdim_actor(setup)
        else:
            log.debug(f'start leaner with queue {rank}')
            train_loop_mfdim_learner(setup, setup.queues[rank])
    except:
        log.exception('Error in training loop')
        raise

    setup.close()


@hydra.main(config_path='config')
def main(cfg: DictConfig):
    log.info(f'** running from source tree at {hydra.utils.get_original_cwd()}')
    log.info(f'** running at {os.getcwd()}')
    log.info(f'** configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    procs = []
    rdvu_file = f'{cfg.distributed.rdvu_path}/rdvu-{uuid.uuid4()}'
    na = int(cfg.distributed.num_actors)
    nl = int(cfg.distributed.num_learners)
    cfg.distributed = {
        'num_actors': na,
        'num_learners': nl,
        'size': na + nl,
        'init_method': f'file://{rdvu_file}',
        'role': None,
    }

    if cfg.agent.batch_size % nl != 0:
        raise ValueError('Batch size must be multiple of num_learners')
    if cfg.agent.samples_per_update % nl != 0:
        raise ValueError('Samples per update must be multiple of num_learners')
    if cfg.agent.warmup_samples % nl != 0:
        raise ValueError('Warmup samples must be multiple of num_learners')
    if cfg.env.train_procs % nl != 0:
        raise ValueError('Train procs should be multiple of num_learners')

    queues = [mp.Queue() for _ in range(nl)]
    bcast_barrier = mp.Barrier(na + nl)
    rank = 0
    for _ in range(nl):
        p = mp.Process(
            target=worker, args=(rank, 'learner', queues, bcast_barrier, cfg)
        )
        procs.append(p)
        rank += 1
    for _ in range(na):
        p = mp.Process(
            target=worker, args=(rank, 'actor', queues, bcast_barrier, cfg)
        )
        procs.append(p)
        rank += 1
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    try:
        os.remove(rdvu_file)
    except:
        pass


if __name__ == '__main__':
    main()
