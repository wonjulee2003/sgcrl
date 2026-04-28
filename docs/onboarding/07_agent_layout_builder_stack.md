# Agent, Layout, Builder Stack

## One-Sentence Summary
`agents.py` turns SGCRL config into a distributed agent, `distributed_layout.py` wires the worker graph, and `builder.py` creates the concrete Acme pieces: actor, learner, replay table, replay writer, and replay dataset.

## Library Cheat Sheet
| Library | Role | Used here for |
|---|---|---|
| Launchpad | process launcher | replay, learner, actors, evaluator, counter workers |
| Acme | RL framework | actors, learners, environment loops, counters, adders |
| Reverb | replay server | stores full episodes and serves learner samples |
| JAX | array/autodiff/JIT | random keys, learner updates, network compute |
| Haiku | JAX neural nets | policy/critic definitions in `networks.py` |
| Optax | JAX optimizers | Adam for policy and critic |
| TensorFlow | data pipeline | replay dataset and goal relabeling pipeline |
| Gym | env API | raw Sawyer/Point env `reset`, `step`, spaces |
| dm_env | Acme env API | wrapped env interface for `EnvironmentLoop` |
| MuJoCo/MetaWorld | 3D sim/tasks | Sawyer robot environments |

Mental split: MuJoCo/MetaWorld/Gym are the environment world; Launchpad/Acme/Reverb are distributed RL infrastructure; JAX/Haiku/Optax are model computation; TensorFlow is the replay input pipeline.

## File Roles
```text
lp_contrastive.py
  -> builds config, env factories, network factory
  -> creates DistributedContrastive

contrastive/agents.py
  -> SGCRL-specific distributed agent adapter
  -> chooses builder, loggers, evaluators, observers

contrastive/distributed_layout.py
  -> generic worker topology
  -> creates Launchpad nodes for replay/counter/learner/evaluator/actors

contrastive/builder.py
  -> concrete Acme builder
  -> creates learner, actor, replay table, dataset iterator, adder
```

## 1. `agents.py`: Config Becomes A Distributed Agent
Main class: `contrastive.agents.DistributedContrastive`.

Inputs from `lp_contrastive.py`:
- `environment_factory`: training env factory.
- `environment_factory_fixed_goals`: evaluation env factory.
- `network_factory`: builds policy/critic networks from env spec.
- `config`: hyperparameters plus `obs_dim`, `max_episode_steps`, goal indices.
- `seed`, `num_actors`, `max_number_of_steps`.

It creates `logger_fn`, `ContrastiveBuilder(config, logger_fn)`, eval/actor observers, evaluator factory, actor logger factory, and checkpoint config.

Then it calls `DistributedLayout.__init__`.

Practical meaning: `agents.py` decides what kind of distributed agent this is; `distributed_layout.py` decides how to run it as workers; `builder.py` decides how each worker's internals are made.

Important details:
- Learner/actor/evaluator logs use `config.log_dir + config.alg_name + '_' + config.env_name + '_' + seed`.
- Evaluator uses `environment_factory_fixed_goals`, even if training actors sample goals.
- If `config.local=True`, evaluator factories are removed.
- Actor policy is `networks.apply_policy_and_sample`.
- Evaluator passes `True` to `apply_policy_and_sample`.

## 2. `distributed_layout.py`: Workers Are Wired
Main class: `contrastive.distributed_layout.DistributedLayout`.

This file is mostly infrastructure. It does not define the SGCRL loss.

Worker methods:
| Method | What it builds |
|---|---|
| `replay` | Reverb replay tables |
| `counter` | checkpointed Acme counter |
| `learner` | checkpointed `ContrastiveLearner` worker |
| `actor` | training `EnvironmentLoop` worker |
| `coordinator` | max actor-step limiter |
| `build` | Launchpad program graph |

Launchpad wiring in `build`:
```text
replay     = lp.ReverbNode(self.replay)
counter    = lp.CourierNode(self.counter)
learner    = lp.CourierNode(self.learner, learner_key, replay, counter)
evaluator  = lp.CourierNode(evaluator, evaluator_key, learner, counter, make_actor)
actor      = lp.CourierNode(self.actor, actor_key, replay, learner, counter, actor_id)
```

The handles are the important part:
- `replay` goes to actors for writing and learner for reading.
- `learner` goes to actors/evaluator as `variable_source`.
- `counter` is shared for logs, checkpoints, and step limiting.

Actor worker: make adder, env, networks, policy, actor, then return `EnvironmentLoop`.

Learner worker: make replay iterator, env spec, networks, learner, then wrap in `CheckpointingRunner`.

Evaluator worker: make fixed-goal env and eval actor using learner as variable source, then return `EnvironmentLoop`.

Ignore first: type aliases, `MultiThreadingColocation`, `device_prefetch`, and Courier internals.

## 3. `builder.py`: Actor, Learner, Replay, Dataset
Main class: `contrastive.builder.ContrastiveBuilder`.

This is an Acme `ActorLearnerBuilder`: it knows how to create each RL component.

### `make_learner`
Creates Adam optimizers and returns `ContrastiveLearner(...)`, passing networks, random key, replay iterator, counter, logger, `obs_to_goal`, and config.

### `make_actor`
Creates the runtime policy actor:
```text
policy_network
  -> batched_feed_forward_to_actor_core
  -> VariableClient(variable_source, 'policy')
  -> InitiallyRandomActor or GenericActor
```

Key idea: `variable_source = learner`; `VariableClient` asks learner for `policy` params; actor uses those params to choose actions.

If `config.use_random_actor=True`, `InitiallyRandomActor` takes uniform random actions until policy params are no longer initial.

### `make_replay_tables`
Creates the Reverb table with uniform sampling, FIFO removal, `SampleToInsertRatio` rate limiting, and an `EpisodeAdder` signature.

Capacity is episode-based: `min/max_replay_size // max_episode_steps`.

### `make_adder`
Creates the actor-side replay writer:
```text
EpisodeAdder(
  client=replay_client,
  priority_fns={replay_table_name: None},
  max_sequence_length=max_episode_steps + 1
)
```

This writes complete episodes, not individual transitions.

### `make_dataset_iterator`
Creates the learner-side replay reader:
```text
Reverb TrajectoryDataset
  -> flatten_fn(sample)
  -> batch episodes
  -> transpose shuffle
  -> unbatch into transitions
  -> batch(batch_size * num_sgd_steps_per_step)
  -> wrap as ReverbSample
  -> as_numpy_iterator()
```

SGCRL-specific part:
```text
state = observation[:-1, :obs_dim]
next_state = observation[1:, :obs_dim]
goal = obs_to_goal_2d(sampled future states)
new_obs = concat([state, goal])
new_next_obs = concat([next_state, goal])
```

So `builder.py` is infrastructure plus the critical replay relabeling step.

## Environment Stack
```text
lp_contrastive.py::get_program
  -> contrastive.utils.make_environment
  -> env_utils.load
  -> MuJoCo / MetaWorld / Gym or PointEnv
  -> Acme wrappers
```

## End-To-End Runtime Story
```text
1. lp_contrastive.py builds config and factories.
2. DistributedContrastive creates builder, loggers, observers, evaluator factory.
3. DistributedLayout.build creates Launchpad nodes.
4. Launchpad starts replay, counter, learner, evaluator, and actors.
5. Actors run envs and write episodes to Reverb.
6. Learner samples episodes, relabels goals, updates params.
7. Actors/evaluator fetch updated policy params from learner.
8. Counter/loggers/checkpointers track progress.
```

## Tiny Glossary
- **Actor**: runs an env, selects actions, writes episodes.
- **Learner**: consumes replay and updates params.
- **Evaluator**: actor-like worker using fixed-goal envs.
- **Variable source**: learner interface that serves params.
- **Variable client**: actor-side object that fetches params.
- **Adder**: writes env episodes to replay.
- **Replay table**: Reverb storage for sampled trajectories.
- **EnvironmentLoop**: Acme loop connecting env and actor.

