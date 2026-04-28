# Goal Flow Short

## One-Sentence Summary

Goals start in environment reset, observations are flat `[state, goal]`, replay later replaces the stored goal with a sampled future-state goal, and the learner splits state from goal using `config.obs_dim`.

## Basic Flow

```text
lp_contrastive.py::main
  -> params['fix_goals'] = not FLAGS.sample_goals
  -> lp_contrastive.py::get_program
  -> contrastive.utils.make_environment
  -> env reset creates [state, goal]
  -> ObservationFilterWrapper exposes [state, selected_goal]
  -> replay stores episodes
  -> ContrastiveBuilder.make_dataset_iterator relabels goals from future states
  -> learner sees [state, relabeled_goal]
```

## 1. Where Goals Enter

Fixed goals are listed in:

```text
lp_contrastive.py::fixed_goal_dict
```

Sampled goals are created during reset:

- `env_utils.SawyerBin.reset`
- `env_utils.SawyerBox.reset`
- `env_utils.SawyerPeg.reset`
- `point_env.PointEnv.reset`

## 2. Observation Structure

The observation is a flat vector:

```text
observation = [state, goal]
```

Raw env references:

- `env_utils.SawyerBin._get_obs`
- `env_utils.SawyerBox._get_obs`
- `env_utils.SawyerPeg._get_obs`
- `point_env.PointEnv._get_obs`

Wrapper reference:

```text
contrastive.utils.ObservationFilterWrapper
```

## 3. What `--sample_goals` Changes

In `lp_contrastive.py::main`:

```text
params['fix_goals'] = not FLAGS.sample_goals
```

So:

```text
--sample_goals False  -> fixed training goals
--sample_goals True   -> environment-sampled training goals
```

Evaluation still uses the fixed-goal factory:

```text
lp_contrastive.py::get_program
  -> env_factory_fixed_goals
```

## 4. Where Goals Are Fixed

Fixed values:

```text
lp_contrastive.py::fixed_goal_dict
```

Passed through:

```text
lp_contrastive.py::get_program
  -> contrastive.utils.make_environment(..., fixed_start_end=...)
  -> env_utils.load
  -> env constructor
```

## 5. Where Goals Are Sampled

Environment sampling:

- `env_utils.SawyerBin.reset`: interpolates/samples `self._goal`
- `env_utils.SawyerBox.reset`: interpolates `self._goal_pos`
- `env_utils.SawyerPeg.reset`: interpolates `self._goal_pos`
- `point_env.PointEnv.reset`: samples `self.goal`

Learner future-state relabeling:

```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn
```

Actor-loss goal mixing:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss
```

## 6. Future-State Relabeling

File/function:

```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn
```

It does:

```text
state = observation[:-1, :obs_dim]
next_state = observation[1:, :obs_dim]
goal candidates = obs_to_goal_2d(observation[:, :obs_dim])
goal_index = sampled future timestep
goal = gather(goal candidates, goal_index[:-1])
new_obs = concat([state, goal])
new_next_obs = concat([next_state, goal])
```

Meaning:

```text
train on [s_t, h(s_k)] where k > t
```

## 7. How Learner Splits State And Goal

The split point is:

```text
config.obs_dim
```

References:

- `lp_contrastive.py::get_program` sets `config.obs_dim`
- `contrastive.learning.ContrastiveLearner.__init__.<locals>.critic_loss`
- `contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss`
- `contrastive.networks.make_networks.<locals>._repr_fn`

Pattern:

```text
state = obs[:, :config.obs_dim]
goal = obs[:, config.obs_dim:]
```

## 8. Evaluator Goals

Evaluator uses fixed goals:

```text
contrastive.agents.DistributedContrastive.__init__
  -> distributed_layout.default_evaluator_factory(
       environment_factory=environment_factory_fixed_goals
     )
```

The evaluator policy uses:

```text
networks.apply_policy_and_sample(n, True)
```

So evaluation uses mode actions and fixed-goal environments.

## Fixed SGCRL vs Original Contrastive RL Style

Fixed SGCRL default:

```text
--sample_goals False
training env goals fixed
evaluation goals fixed
```

Original contrastive RL-style:

```text
--sample_goals True
training env goals sampled
evaluation goals still fixed in this repo
```

Both still use learner future-state relabeling.

## Open Questions

1. `env_utils.SawyerBox.reset` and `env_utils.SawyerPeg.reset` have fixed branches but do not visibly assign `self._fixed_start_end` to the goal.
2. `--sample_goals` changes training goals, but evaluator goals remain fixed.
3. Future-state relabeling discards stored environment goals and uses stored states as candidate goals.

