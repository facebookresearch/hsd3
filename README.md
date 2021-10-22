# Hierarchical Skills for Efficient Exploration

This is the source code release for the paper [Hierarchical Skills for Efficient
Exploration](https://arxiv.org/abs/2110.10809). It contains

- Code for pre-training and hierarchical learning with HSD-3
- Code for the baselines we compare to in the paper

Additionally, we provide pre-trained skill policies for the Walker and Humanoid
robots considered in the paper.

The benchmark suite can be found in a standlone repository at
[facebookresearch/bipedal-skills](https://github.com/facebookresearch/bipedal-skills)

## Prerequisites

Install PyTorch according to the [official
instructions](https://pytorch.org/get-started), for example in a new conda
environment. This code-base was tested with PyTorch 1.8 and 1.9.

Then, install remaining requirements via
```sh
pip install -r requirements.txt
```

For optimal performance, we also recommend installing NVidia's
[PyTorch extensions](https://github.com/NVIDIA/apex).


## Usage

We use [Hydra](https://hydra.cc) to handle training configurations, with some defaults that might
not make everyone happy. In particular, we disable the default job directory
management -- which is good for local development but not desirable for running
full experiments. This can be changed by adapting the initial portion of
`config/common.yaml` or by passing something like
`hydra.run.dir=./outputs/my-custom-string` to the commands below.

### Pre-training Hierarchical Skills

For pre-training skill policies, use the `pretrain.py` script (note that this
requires a machine with 2 GPUs):
```sh
# Walker robot
python pretrain.py -cn walker_pretrain
# Humanoid robot
python pretrain.py -cn humanoid_pretrain
```

### Hierarchical Control

High-level policy training with HSD-3 is done as follows:
```sh
# Walker robot
python train.py -cn walker_hsd3
# Humanoid robot
python train.py -cn humanoid_hsd3
```
The default configuration assumes that a pre-trained skill policy is available
at `checkpoint-lo.pt`. The location can be overriden by setting a new value for
`agent.lo.init_from` (see below for an example). By default, a high-level agent
will be trained on the "Hurdles" task. This can be changed by passing
`env.name=BiskStairs-v1`, for example.

Pre-trained skill policies are available
[here](https://dl.fbaipublicfiles.com/hsd3/pretrained-skills.tar.gz). After
unpacking the archive in the top-level directory of this repository, they can
be used as follows:
```sh
# Walker robot
python train.py -cn walker_hsd3 agent.lo.init_from=$PWD/pretrained-skills/walker.pt
# Humanoid robot
python train.py -cn humanoid_hsd3 agent.lo.init_from=$PWD/pretrained-skills/humanoidpc.pt
```

### Baselines

Individual baselines can be run by passing the following as the `-cn` argument to `train.py` (for the Walker robot):

Baseline | Configuration name
--- | ---
Soft Actor-Critic | `walker_sac`
DIAYN-C pre-training | `walker_diaync_pretrain`
DIAYN-C HRL | `walker_diaync_hrl`
HIRO-SAC | `walker_hiro`
Switching Ensemble | `walker_se`
HSD-Bandit | `walker_hsdb`
SD | `walker_sd`

By default, `walker_sd` will select the full goal space. Other goal spaces can
be selected by modifying the configuration, e.g., passing `subsets=2-3+4` will
limit high-level control to X translation (2) and the left foot (3+4).


## License
hsd3 is MIT licensed, as found in the LICENSE file.
