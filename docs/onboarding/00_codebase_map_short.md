# Codebase Map Short

## One-Paragraph Overview

This repo implements Single-goal Contrastive RL using JAX, Haiku, Acme, Reverb, and Launchpad. `lp_contrastive.py::main` parses flags, creates a `contrastive.config.ContrastiveConfig`, builds environments through `contrastive.utils.make_environment`, builds networks through `contrastive.networks.make_networks`, and launches a distributed training program through `contrastive.agents.DistributedContrastive`.

## Main Execution Path

```text
lp_contrastive.py::main
  -> lp_contrastive.py::get_program
  -> contrastive.config.ContrastiveConfig
  -> contrastive.utils.make_environment
  -> contrastive.networks.make_networks
  -> contrastive.agents.DistributedContrastive
  -> contrastive.distributed_layout.DistributedLayout.build
  -> replay / learner / actors / evaluator
```

## Core Files

| File | Why it matters | Category |
|---|---|---|
| `lp_contrastive.py` | CLI entry point; maps flags to config and launches training. | infrastructure |
| `contrastive/config.py` | Defines `contrastive.config.ContrastiveConfig`; most behavior is config-driven. | infrastructure |
| `contrastive/agents.py` | Defines `contrastive.agents.DistributedContrastive`; assembles the agent system. | infrastructure |
| `contrastive/distributed_layout.py` | Launchpad graph: replay, learner, actors, evaluator, counter. | infrastructure |
| `contrastive/builder.py` | Creates learner, actor, replay table, dataset iterator, adder. | infrastructure |
| `contrastive/learning.py` | Main losses and parameter updates in `contrastive.learning.ContrastiveLearner`. | algorithm |
| `contrastive/networks.py` | Policy and critic architecture in `contrastive.networks.make_networks`. | algorithm |
| `contrastive/utils.py` | Env wrapping, goal extraction, observers, random-initial actor. | environment/utility |
| `env_utils.py` | Sawyer task wrappers and env loader. | environment |
| `point_env.py` | 2D point navigation environment. | environment |
| `distributional.py` | Tanh-normal action distribution used by the policy. | utility |
| `default.py` | Logger factory used by learner, actors, evaluator. | utility |

## Top Things To Understand First

1. `lp_contrastive.py::main`: reads `--env`, `--alg`, `--sample_goals`, `--num_steps`.
2. `lp_contrastive.py::get_program`: builds config, environments, networks, and `DistributedContrastive`.
3. `contrastive.config.ContrastiveConfig`: holds `batch_size`, `discount`, `repr_dim`, `use_td`, `use_cpc`, `twin_q`, etc.
4. `contrastive.builder.ContrastiveBuilder.make_dataset_iterator`: turns replay episodes into future-goal relabeled learner batches.
5. `contrastive.networks.make_networks`: builds `policy_network` and `q_network`.
6. `contrastive.learning.ContrastiveLearner.__init__`: defines `critic_loss`, `actor_loss`, and `update_step`.
7. `contrastive.learning.ContrastiveLearner.step`: pulls replay batch, updates params, logs metrics.

## Mental Model

```text
actors collect episodes
  -> Reverb replay stores them
  -> learner samples and relabels future goals
  -> critic learns state-action vs goal compatibility
  -> actor/policy learns actions that score high for goals
```

## Read First

- `lp_contrastive.py::main`
- `lp_contrastive.py::get_program`
- `contrastive/builder.py::ContrastiveBuilder.make_dataset_iterator`
- `contrastive/networks.py::make_networks`
- `contrastive/learning.py::ContrastiveLearner.__init__`

## Read Later

- `contrastive/distributed_layout.py::DistributedLayout.build`
- `contrastive/agents.py::DistributedContrastive.__init__`
- `env_utils.py::SawyerBin`, `env_utils.py::SawyerBox`, `env_utils.py::SawyerPeg`
- `point_env.py::PointEnv`
- `distributional.py::NormalTanhDistribution`

## Probably Ignore Initially

- Unused distribution heads in `distributional.py`
- Logging details in `default.py`
- `training_videos/`
- Unused imports and unused config fields

## Open Questions

1. `contrastive.networks.make_networks.<locals>._critic_fn` references `product` in the `twin_q=True` branch, but the visible first matrix is named `critic_val`.
2. `contrastive_nce` is listed in `README.md`, but `lp_contrastive.py::main` does not handle it.
3. `env_utils.SawyerBox.reset` and `env_utils.SawyerPeg.reset` have fixed-goal branches that do not directly use `self._fixed_start_end`.

