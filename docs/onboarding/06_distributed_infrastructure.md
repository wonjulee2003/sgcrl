# Distributed Infrastructure

## One-Sentence Summary
Launchpad starts the workers, Acme gives them standard RL interfaces, Reverb stores actor episodes, and SGCRL-specific learning happens when replay episodes are future-goal relabeled and consumed by `ContrastiveLearner`.

## Basic Flow
```text
lp_contrastive.py::main
  -> lp_contrastive.py::get_program
  -> contrastive.agents.DistributedContrastive
  -> contrastive.distributed_layout.DistributedLayout.build
  -> replay / counter / learner / evaluator / actor workers
```

Repo-specific diagram:
```text
environment actors -> replay -> learner -> updated policy -> actors/evaluator
        |              |         |               |              |
        |              |         |               |              + fixed-goal eval env
        |              |         |               + learner.get_variables('policy')
        |              |         + ContrastiveLearner.step updates policy/q params
        |              + Reverb table of whole episodes
        + EnvironmentLoop(policy actor, env, observers, EpisodeAdder)
```

## 1. What Launchpad Is Doing
Launchpad is the process launcher and handle-wiring layer.

Program nodes:
```text
replay     -> lp.ReverbNode(self.replay)
counter    -> lp.CourierNode(self.counter)
learner    -> lp.CourierNode(self.learner, learner_key, replay, counter)
evaluator  -> lp.CourierNode(evaluator, evaluator_key, learner, counter, make_actor)
actor      -> lp.CourierNode(self.actor, actor_key, replay, learner, counter, actor_id)
```

Must understand:
- Launchpad does not define SGCRL math.
- `replay` is the Reverb server handle.
- `learner` is the variable source for policy params.
- `counter` is shared by learner, actors, evaluator, and step limiter.

Ignore first:
- Courier internals.
- Launchpad launch types.
- `lp.MultiThreadingColocation`.

## 2. What Acme Is Doing
Acme supplies the RL plumbing around this repo's code. Main references:
- `contrastive.builder.ContrastiveBuilder`
- `contrastive.distributed_layout.default_evaluator_factory`
- `acme.environment_loop.EnvironmentLoop`

Mapping: builder = `ContrastiveBuilder`; learner = `ContrastiveLearner`; actor = `GenericActor` or `InitiallyRandomActor`; variable source = learner handle; adder = `EpisodeAdder`; loop = actor/evaluator `EnvironmentLoop`.

Must understand:
- Actors fetch `policy` variables from the learner.
- Adders write trajectories to replay.
- `EnvironmentLoop` connects env reset/step to actor select-action.

Ignore first:
- Generic Acme APIs outside this repo.
- Bigtable naming in logger arguments.

## 3. What Reverb / Replay Is Doing
Reverb is the episode replay server. Main references:
- `ContrastiveBuilder.make_replay_tables`
- `ContrastiveBuilder.make_adder`
- `ContrastiveBuilder.make_dataset_iterator`

Table setup:
```text
reverb.Table(
  sampler=Uniform(),
  remover=Fifo(),
  max_size=max_replay_size // max_episode_steps,
  rate_limiter=SampleToInsertRatio(...),
  signature=EpisodeAdder.signature(environment_spec, {})
)
```

Actors insert whole episodes:
```text
EpisodeAdder(
  client=replay_client,
  max_sequence_length=config.max_episode_steps + 1
)
```

Learner reads:
```text
TrajectoryDataset.from_table_signature(...)
  -> flatten_fn
  -> future-goal relabeling
  -> batch(batch_size * num_sgd_steps_per_step)
```

SGCRL-critical detail:
```text
stored actor observation   = [state, environment_goal]
learner replay observation = [state, sampled_future_state_goal]
```

Replay stores raw episodes. `flatten_fn` decides the training goals.

## 4. What The Actor Process Does
Path:
```text
DistributedLayout.actor
  -> builder.make_adder(replay)
  -> environment_factory(seed)
  -> network_factory(environment_spec)
  -> networks.apply_policy_and_sample(networks)
  -> builder.make_actor(actor_key, policy_network, adder, learner)
  -> EnvironmentLoop(environment, actor, counter, logger, observers)
```

Responsibilities:
- Run the training environment.
- Select actions from the current policy.
- Use uniform random actions initially when `config.use_random_actor=True`.
- Insert complete episodes into Reverb.
- Log actor steps, success, and distance metrics.

## 5. What The Learner Process Does
Path:
```text
DistributedLayout.learner
  -> builder.make_dataset_iterator(replay)
  -> network_factory(environment_spec)
  -> builder.make_learner(...)
  -> savers.CheckpointingRunner(learner)
```

Step:
```text
ContrastiveLearner.step
  -> sample = next(iterator)
  -> transitions = acme.types.Transition(*sample.data)
  -> update critic
  -> update target critic
  -> update actor policy
  -> counter.increment(...)
  -> logger.write(...)
```

Learner-owned state:
- `policy_params`
- `q_params`
- `target_q_params`
- policy/q optimizer states
- `policy_params_prev`
- RNG key
- optional entropy-alpha state

Served variables:
```text
ContrastiveLearner.get_variables(['policy', 'critic'])
```

Actors and evaluator normally fetch only `policy`.

## 6. What The Evaluator Does
Path:
```text
DistributedContrastive.__init__
  -> default_evaluator_factory(environment_factory_fixed_goals, ...)
  -> EnvironmentLoop(fixed_goal_environment, eval_actor, counter, logger, observers)
```

Important details:
- Uses `environment_factory_fixed_goals`, regardless of `--sample_goals`.
- Policy factory is `networks.apply_policy_and_sample(n, True)`.
- Observers are `SuccessObserver` and `DistanceObserver`.
- Disabled when `config.local=True`.

## 7. What Gets Checkpointed And Logged
Checkpointed:
- Learner via `savers.CheckpointingRunner(..., key='learner', subdirectory='learner')`.
- Counter via `savers.CheckpointingRunner(..., key='counter', subdirectory='counter')`.

Learner checkpoint contents: `ContrastiveLearner.save -> TrainingState`, including policy params, critic params, target critic params, optimizer states, previous policy params, RNG key, and optional alpha state.

Logged:
- Learner: `critic_loss`, `actor_loss`, accuracies, logits, entropy, `steps_per_second`, counter values.
- Actor: actor steps plus success/distance observer outputs.
- Evaluator: fixed-goal success/distance plus evaluator counter values.

Logger/path references: `default.make_default_logger`; `config.log_dir + config.alg_name + '_' + config.env_name + '_' + str(seed)`.

## 8. Essential SGCRL vs Infrastructure
Essential to SGCRL:
- `[state, goal]` observation convention.
- `config.obs_dim` state/goal split.
- Future-state goal relabeling in `make_dataset_iterator`.
- Contrastive critic logits over state-action and goal representations.
- Actor objective maximizing the diagonal critic score.
- Fixed-goal evaluator for this repo's SGCRL runs.

Infrastructure:
- Launchpad process groups.
- Acme `EnvironmentLoop`, `Builder`, `VariableClient`, counters.
- Reverb tables, adders, and rate limiter.
- Checkpointing runners.
- Logger wrappers.
- Prefetch and colocation.

## First-Onboarding Advice
Must understand:
1. Actors collect episodes; they do not train.
2. Replay stores episodes; it does not choose final learner goals.
3. `flatten_fn` relabels goals from future states.
4. The learner is the only process updating policy and critic params.
5. Actors/evaluator fetch fresh policy params from the learner handle.

Safe to ignore first:
- Launchpad/Courier internals.
- Reverb rate-limiter math.
- `lp_launch_type` choices.
- Bigtable wording.
- Device prefetch and multi-thread colocation.
- Checkpoint UID details.

## Practical Mental Model
```text
Launchpad = runs workers
Acme      = actor/learner/replay interfaces
Reverb    = episode storage
SGCRL     = replay relabeling + contrastive learner update
```

For algorithm behavior, start in:
```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator
contrastive.learning.ContrastiveLearner
contrastive.networks.make_networks
```

For process wiring, start in:
```text
contrastive.distributed_layout.DistributedLayout.build
contrastive.distributed_layout.DistributedLayout.actor
contrastive.distributed_layout.DistributedLayout.learner
```

