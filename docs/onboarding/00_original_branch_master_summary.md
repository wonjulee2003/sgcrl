# Original Branch Master Summary
## Scope
Portable baseline summary of the original episodic/distributed SGCRL branch, for comparing against another codebase or a non-episodic branch.
## 1. Purpose
This repo implements Single-goal Contrastive RL using JAX/Haiku/Optax for computation, Acme/Reverb for actor-learner replay infrastructure, Launchpad for distributed execution, and Gym/MetaWorld/MuJoCo plus `point_env.py` for environments.

Core loop:
```text
actors collect goal-conditioned episodes -> Reverb stores episodes
  -> learner samples episodes and relabels goals with future states
  -> critic learns state-action/goal compatibility
  -> actor learns actions that score highly for goals
```
Core algorithm: `contrastive/builder.py::ContrastiveBuilder.make_dataset_iterator`, `contrastive/learning.py::ContrastiveLearner`, `contrastive/networks.py::make_networks`.

Core infrastructure: `lp_contrastive.py`, `contrastive/agents.py`, `contrastive/distributed_layout.py`, `contrastive/builder.py`.
## 2. Main Execution Path
```text
lp_contrastive.py::main
  -> parse --env/--alg/--num_steps/--sample_goals/--seed
  -> params['fix_goals'] = not FLAGS.sample_goals
  -> lp_contrastive.py::get_program(params)
  -> contrastive.config.ContrastiveConfig(**params)
  -> contrastive.utils.make_environment(...) -> env_utils.load(...)
  -> set config.obs_dim and config.max_episode_steps
  -> network_factory = functools.partial(contrastive.make_networks, ...)
  -> contrastive.agents.DistributedContrastive(...)
  -> contrastive.distributed_layout.DistributedLayout.build()
  -> lp.launch(program, terminal='current_terminal')
```
`agent.build()` only builds the Launchpad `Program`; `lp.launch(...)` executes it.

Runtime graph:
```text
replay / counter / learner / evaluator / actors
environment actors -> Reverb replay -> learner -> updated policy -> actors/evaluator
```
## 3. Ten Most Important Files
| File | Role |
|---|---|
| `lp_contrastive.py` | CLI entry point; builds config/factories/agent and calls `lp.launch`. |
| `contrastive/config.py` | `ContrastiveConfig`; batch, replay, algorithm, env, entropy, distributed settings. |
| `contrastive/agents.py` | `DistributedContrastive`; packages config into builder, evaluator, observers, loggers. |
| `contrastive/distributed_layout.py` | `DistributedLayout`; Launchpad graph and worker factories. |
| `contrastive/builder.py` | `ContrastiveBuilder`; actor, learner, replay table, adder, dataset iterator. |
| `contrastive/learning.py` | `ContrastiveLearner`; losses, updates, metrics, variables, save/restore. |
| `contrastive/networks.py` | Policy/critic/repr networks and action sampling helpers. |
| `contrastive/utils.py` | Env wrappers, goal slicing, observers, `InitiallyRandomActor`. |
| `env_utils.py` | `SawyerBin`, `SawyerBox`, `SawyerPeg`, `load`; MetaWorld/MuJoCo wrappers. |
| `point_env.py` | Local 2D `PointEnv`. |
Support: `distributional.py::NormalTanhDistribution`, `default.py::make_default_logger`, `contrastive/__init__.py`.
## 4. Original Learner Update Path
Public learner step:
```text
ContrastiveLearner.step
  -> sample = next(self._iterator)
  -> transitions = acme.types.Transition(*sample.data)
  -> self._state, metrics = self._update_step(self._state, transitions)
  -> counter increment + logger write
```
Iterator/learner creation: `ContrastiveBuilder.make_dataset_iterator`, `ContrastiveBuilder.make_learner`.

Default batch: `B = 256`, `N = config.num_sgd_steps_per_step = 64`, `B_total = 16384`. `ContrastiveLearner.__init__` wraps `update_step` with `acme.jax.utils.process_multiple_batches(update_step, N)`.

Update sequence:
```text
replay batch -> Transition
  -> critic_loss(...)
  -> q_optimizer.update / optax.apply_updates
  -> target_q_params = (1 - tau) * target_q_params + tau * q_params
  -> actor_loss(...)
  -> policy_optimizer.update / optax.apply_updates
  -> metrics
```
Critic branches in `ContrastiveLearner.__init__.<locals>.critic_loss`:
- `use_td=False`, `use_cpc=True`: CPC softmax cross-entropy over `[B, B]`.
- `use_td=False`, `use_cpc=False`: sigmoid BCE/NCE-style branch.
- `use_td=True`: TD/C-learning branch with `target_q_params`, next actions, twin critic.

Actor branch in `ContrastiveLearner.__init__.<locals>.actor_loss`:
```text
action ~ pi(. | state, goal)
q_action = Q(state, action, goal)
actor_loss = mean(-diag(q_action))
```
Entropy term exists as `- alpha * (-log_prob)`, but `lp_contrastive.py::main` sets `entropy_coefficient = 0.0`, so default entry-point runs give it zero weight.

Main metrics: `critic_loss`, `actor_loss`, `binary_accuracy`, `categorical_accuracy`, `logits_pos`, `logits_neg`, `logsumexp`, `entropy_mean`, `steps_per_second`.
## 5. Goals In The Original Branch
Observation convention:
```text
observation = [state, goal]
state = obs[:, :config.obs_dim]
goal  = obs[:, config.obs_dim:]
```
Used by `ContrastiveLearner.critic_loss`, `ContrastiveLearner.actor_loss`, `contrastive.networks.make_networks.<locals>._repr_fn`, and `contrastive.utils.DistanceObserver._get_distance`.

Default dimensions: `point_Spiral11x11` is `2+2=4`; `sawyer_bin` is `7+7=14`; `sawyer_box` is `11+11=22`; `sawyer_peg` is `7+7=14`.

Fixed goals live in `lp_contrastive.py::fixed_goal_dict`.

Training:
```text
--sample_goals False -> fixed training goals
--sample_goals True  -> env-sampled training goals
```
Evaluation always uses `lp_contrastive.py::get_program -> env_factory_fixed_goals`.

Environment goal reset locations: `env_utils.SawyerBin.reset`, `env_utils.SawyerBox.reset`, `env_utils.SawyerPeg.reset`, `point_env.PointEnv.reset`.

Learner relabeling in `ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn` discards stored env goals for training targets:
```text
goal candidates = obs_to_goal_2d(sample.data.observation[:, :config.obs_dim])
goal = gather(candidate future-state goals)
```
Actor loss also mixes goals via `config.random_goals`: `0.0` original, `0.5` original plus rolled goals, `1.0` rolled only.
## 6. Replay And Dataset Iteration
Replay table:
```text
ContrastiveBuilder.make_replay_tables
  -> reverb.Table(..., sampler=Uniform(), remover=Fifo(),
       max_size=max_replay_size // max_episode_steps,
       rate_limiter=SampleToInsertRatio(...),
       signature=EpisodeAdder.signature(environment_spec, {}))
```
Actor writer:
```text
ContrastiveBuilder.make_adder
  -> adders_reverb.EpisodeAdder(..., max_sequence_length=config.max_episode_steps + 1)
```
This is episode/sequence replay, not independent transition-row replay.

Dataset:
```text
TrajectoryDataset.from_table_signature(...)
  -> map(flatten_fn)
  -> batch(batch_size) -> transpose shuffle -> unbatch twice
  -> batch(batch_size * num_sgd_steps_per_step)
  -> wrap as reverb.ReplaySample
```
`flatten_fn` receives one trajectory and creates:
```text
observation       = concat([state_t, h(state_k)])
action            = action_t
reward            = reward_t
discount          = sample.data.discount_t
next_observation  = concat([state_{t+1}, h(state_k)])
extras['next_action'] = action_{t+1}
```
Future goal sampling:
```text
is_future_mask[t, k] = t < k
probs[t, k] = is_future_mask[t, k] * config.discount ** (k - t)
```
So each source timestep samples a strictly future timestep from the same stored sequence. Current fields use `[:-1]`; next fields use `[1:]`.
## 7. Actor/Critic Networks
Factory:
```text
contrastive.networks.make_networks -> ContrastiveNetworks
```
Policy:
```text
_actor_fn([state, goal])
  -> hk.nets.MLP(hidden_layer_sizes), default (256, 256)
  -> distributional.NormalTanhDistribution(action_dim)
```
Critic:
```text
_repr_fn(obs, action)
  -> state = obs[:, :obs_dim]
  -> goal = obs[:, obs_dim:]
  -> sa_repr = MLP(concat([state, action]))
  -> g_repr = MLP(goal)
_combine_repr(sa_repr, g_repr) = einsum('ik,jk->ij', ...)
```
Output is `logits [B, B]`, plus `sa_repr` and `g_repr`. If `twin_q=True`, code intends `[B, B, 2]`, but see open question about `product`.
## 8. Meaning Of The `B x B` Matrix
```text
row i    = phi(s_i, a_i)
column j = psi(g_j)
logits[i, j] = dot(phi(s_i, a_i), psi(g_j))
I = eye(B)
```
Diagonal entries are matched positives; off-diagonal entries are in-batch negatives. CPC uses softmax CE against `I`; NCE-style branch uses sigmoid BCE against `I`; actor loss maximizes the diagonal score for sampled policy actions.
## 9. Episodic Assumptions
- Actors/evaluator are Acme `EnvironmentLoop`s from `DistributedLayout.actor` and `default_evaluator_factory`.
- Raw resets are `SawyerBin.reset`, `SawyerBox.reset`, `SawyerPeg.reset`, `PointEnv.reset`; reset returns `[state, goal]`.
- `contrastive.utils.make_environment` wraps raw envs with `gym_wrapper.GymWrapper`, `step_limit.StepLimitWrapper`, and `ObservationFilterWrapper`.
- Raw `step` methods return `done = False`; episode boundaries come from Acme `StepLimitWrapper`.
- `env_utils.load` sets step limits: `sawyer_* = 150`, `point_* 11x11 = 100`, other point envs `50`.
- `lp_contrastive.py::get_program` sets `config.max_episode_steps = getattr(environment, '_step_limit') + 1`.
- `EpisodeAdder(max_sequence_length=config.max_episode_steps + 1)` encodes a finite sequence assumption; the double `+1` relationship should be checked when porting.
- Replay transition discount is preserved as `sample.data.discount[:-1]`; non-TD CPC loss does not use it.
- `config.discount` weights future goal sampling and TD/C-learning terms.
- Future goals are strictly within the same replay sequence: `t < k`.
- Evaluator uses fixed-goal envs, eval-mode policy, `SuccessObserver`, and `DistanceObserver`; it does not write replay or update params.
- Exact terminal/truncation discount semantics are from Acme wrappers, not local code.
## 10. Files Most Likely To Change In A Non-Episodic Version
- `contrastive/builder.py`: `make_replay_tables`, `make_adder`, `make_dataset_iterator`; episode storage and future-state relabeling live here.
- `contrastive/learning.py`: `ContrastiveLearner.step`, `critic_loss`, `actor_loss`; transition/loss assumptions may change.
- `contrastive/utils.py`: `make_environment`, `ObservationFilterWrapper`, `DistanceObserver`; observation/goal/reset assumptions may change.
- `env_utils.py` and `point_env.py`: env reset, reward, done/truncation, and goal behavior may change.
- `lp_contrastive.py` and `contrastive/config.py`: flags/config for sequence length, replay, discount, goal mode.
- `contrastive/distributed_layout.py` and `contrastive/agents.py`: only if Acme `EnvironmentLoop`, evaluator, or worker topology changes.
- `contrastive/networks.py`: only if `[state, goal]` input contract changes.
## 11. Comparison Checklist For A Non-Episodic Branch
- Does replay still use `adders_reverb.EpisodeAdder`, or transition/window storage?
- Does `make_dataset_iterator` still expect full sequences and use `[:-1]` / `[1:]`?
- What replaces “future state in the same episode” if there are no episodes?
- Is future sampling still weighted by `config.discount ** (k - t)`?
- Are stored environment goals still discarded for learner targets?
- Do raw envs still return `done=False`, and is `StepLimitWrapper` still the only truncation source?
- Is observation still flat `[state, goal]` with the same `obs_dim` split?
- Does `critic_loss` still construct a `[B, B]` contrastive matrix with diagonal positives?
- Does TD/C-learning still use `target_q_params` and `config.discount` the same way?
- Are actors still Acme `EnvironmentLoop`s with `VariableClient(variable_source, 'policy')`?
- Does evaluator still use `environment_factory_fixed_goals`, success, and distance metrics?
- Are logs/checkpoints still `savers.CheckpointingRunner`-managed?
## 12. Open Questions
1. `contrastive.networks.make_networks.<locals>._critic_fn` uses `jnp.stack([product, product2], ...)`, but the first matrix is named `critic_val`.
2. `lp_contrastive.py::main` mentions `contrastive_nce`, but flag dispatch does not accept it.
3. `env_utils.SawyerBox.reset` and `env_utils.SawyerPeg.reset` receive `fixed_start_end`, but fixed branches set `self._goal_pos = pos1` rather than directly using `self._fixed_start_end`.
4. `ContrastiveLearner.__init__.<locals>.update_step` computes updated `q_params`, but `actor_grad` is called with `state.q_params`.
5. `flatten_fn` computes `goal_index` for the final row with no future timestep, then drops it with `goal_index[:-1]`; exact TensorFlow behavior for that all-zero probability row should be verified.
6. Exact `StepLimitWrapper` truncation/discount semantics and `process_multiple_batches` aggregation behavior are external to this repo.
7. `transition.extras['next_action']` is created by `make_dataset_iterator`, but inspected learner losses do not use it.
8. `config.resample_neg_actions`, `config.reward_scale`, `config.n_step`, and `config.no_repr` exist but are not visibly used in the main inspected learner update path.

