# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

log = logging.getLogger(__name__)


# Delta features for standard joint observations
g_delta_feats = {
    'Walker': [1],
    'Humanoid': [0, 1],
    'HumanoidPC': [0, 1],
}


def def_ranges(
    inp: List[Tuple],
    delta_feats: List[int] = None,
    twist_feats: List[int] = None,
) -> Dict[str, List]:
    return {
        'str': [g[1] for g in inp],
        'min': [g[2] for g in inp],
        'max': [g[3] for g in inp],
        'delta_feats': delta_feats if delta_feats else [],
        'twist_feats': twist_feats if twist_feats else [],
    }


g_goal_ranges_bodyfeet_walker: List[Tuple] = [
    (0, 'rootz:p', +0.95, +1.50),  # 0.9 is fall-over
    (1, 'rootx:p', -3.00, +3.00),
    (2, 'rooty:p', -1.30, +1.30),  # 1.4 is fall-over
    (3, 'left_foot:px', -0.72, +0.99),
    (4, 'left_foot:pz', -1.30, +0.00),
    (5, 'right_foot:px', -0.72, +0.99),
    (6, 'right_foot:pz', -1.30, +0.00),
]

g_goal_ranges_bodyfeet_humanoid: List[Tuple] = [
    (0, 'root:px', -3.00, +3.00),
    (1, 'root:py', -3.00, +3.00),  # we can rotate so equalize this
    (2, 'root:pz', +0.95, +1.50),  # 0.9 is falling over
    (3, 'root:tz', -1.57, +1.57),  # stay within +- pi/2, i.e. don't turn around
    (4, 'root:sy', -1.57, +1.57),  # Swing around Y axis
    (5, 'root:sx', -0.50, +0.50),  # Swing around X axis (more or less random v)
    (6, 'left_foot:px', -1.00, +1.00),
    (7, 'left_foot:py', -1.00, +1.00),
    (8, 'left_foot:pz', -1.00, +0.20),
    (9, 'right_foot:px', -1.00, +1.00),
    (10, 'right_foot:py', -1.00, +1.00),
    (11, 'right_foot:pz', -1.00, +0.20),
]

g_goal_spaces_bodyfeet: Dict[str, Dict[str, List]] = {
    'Walker': def_ranges(g_goal_ranges_bodyfeet_walker, [1]),
    'Humanoid': def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
    'HumanoidPC': def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
}

g_goal_spaces: Dict[str, Dict[str, Dict[str, List]]] = {
    'bodyfeet': g_goal_spaces_bodyfeet,
    'bodyfeet-relz': g_goal_spaces_bodyfeet,
}


def subsets_task_map(
    features: str, robot: str, spec: str, rank_min: int, rank_max: int
):
    '''
    Parses a spec of features and returns the necessary data to construct goal
    spaces.
    Spec can be any of the following:
    - 'all': all features
    - 'torso': everything involving the robot's torso/root
    - #-separated list (elements can be feature combinations, i.e. 1+2 or 1-2)
    - a regex matching feature names

    Uncontrollable features will be removed. What's returned is a list of
    feature subsets, each entry in the form "0,1,3+4" (i.e. comma-separated,
    with combinations retained) and a task map that maps each feature to an
    index.
    '''
    gs = g_goal_spaces[features][robot]
    n = len(gs['str'])
    dims: List[str] = []
    if spec == 'all':
        dims = [str(i) for i in range(n)]
    elif spec == 'torso':
        dims = [
            str(i)
            for i in range(n)
            if gs['str'][i].startswith(':')
            or gs['str'][i].startswith('torso:')
            or gs['str'][i].startswith('root')
        ]
    else:
        try:
            dims = []
            for d in str(spec).split('#'):
                ds = sorted(map(int, re.split('[-+]', d)))
                dims.append('+'.join(map(str, ds)))
        except:
            dims = [
                str(i)
                for i in range(n)
                if re.match(spec, gs['str'][i]) is not None
            ]

    def is_controllable(d: int):
        if d < 0:
            raise ValueError(f'Feature {d} out of range')
        if d >= len(gs['min']):
            return False
        # Return whether range is non-zero
        return gs['min'][d] != gs['max'][d]

    uncontrollable = set()
    for dim in dims:
        for idx in map(int, dim.split('+')):
            if not is_controllable(idx):
                uncontrollable.add(dim)
                log.warning(f'Removing uncontrollable feature {dim}')
                break
    dims = [dim for dim in dims if not dim in uncontrollable]
    if len(dims) < rank_min:
        raise ValueError('Less features to control than the requested rank')

    udims: Set[int] = set()
    for dim in dims:
        for idx in map(int, dim.split('+')):
            udims.add(idx)
    task_map: Dict[str, int] = {}
    for idx in sorted(udims):
        task_map[str(idx)] = len(task_map)

    def unify(comb) -> str:
        udims: Set[str] = set()
        for c in comb:
            for d in c.split('+'):
                if d in udims:
                    raise ValueError(f'Overlapping feature dimensions: {comb}')
                udims.add(d)
        return ','.join(
            sorted(comb, key=lambda x: [int(i) for i in x.split('+')])
        )

    if rank_min > 0 and rank_max > 0:
        cdims = []
        for r in range(rank_min, rank_max + 1):
            for comb in combinations(dims, r):
                # XXX Duplications are ok now
                # cdims.append(unify(comb))
                cdims.append(','.join(comb))
        dims = cdims

    return dims, task_map
