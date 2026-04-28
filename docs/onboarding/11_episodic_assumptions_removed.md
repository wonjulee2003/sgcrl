# Episodic Assumptions Removed

## Scope
This note compares current branch `non-episodic` against `main` and focuses only on episode assumptions.

Baseline guide: `main:docs/onboarding/00_original_branch_master_summary.md`.

Diff commands used:

```bash
git diff --name-status main...HEAD
git diff main...HEAD -- contrastive/utils.py env_utils.py contrastive/builder.py lp_contrastive.py
```

Core finding: `contrastive/learning.py`, `contrastive/distributed_layout.py`, and `contrastive/agents.py` are unchanged in `main...HEAD`. The non-episodic change is mostly implemented by changing what the environment reports as episode boundaries.

## 1. Episode Bounds
Original behavior:

- `main:contrastive/utils.py::make_environment` wraps envs with `step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)`.
- Original reference: `main:contrastive/utils.py:162-188`.
- `env_utils.load()` sets finite task limits: Sawyer envs `150`, point `11x11` envs `100`, other point envs `50`.

Non-episodic behavior:

- `contrastive/utils.py::make_environment` now uses `step_limit.StepLimitWrapper(env, step_limit=12_000_000)`.
- Then it adds `FakeEpisodeBoundaryWrapper(env, steps_per_chunk=max_episode_steps)`.
- Current refs: `contrastive/utils.py:356-393`, `env_utils.py:35-66`.

Why it matters:

- The real wrapper-level step limit is no longer the original episode length.
- Original episode length becomes a replay chunk length, not a physical reset horizon.

Status: definitely non-episodic.

## 2. Environment Resets
Original behavior:

- After each `StepLimitWrapper` episode, Acme `EnvironmentLoop` would call reset and the underlying environment would reset.
- Sawyer reset methods recomputed task goals on reset.
- Original refs: `main:env_utils.py:82-99`, `main:env_utils.py:151-167`, `main:env_utils.py:224-238`.

Non-episodic behavior:

- `FakeEpisodeBoundaryWrapper.reset()` calls the real underlying `env.reset()` only when `_needs_reset` is true.
- Later resets return a synthetic `dm_env.TimeStep(step_type=FIRST, reward=0.0, discount=1.0, observation=self._last_observation)`.
- Sawyer wrappers add `_goal_set_once` and only set goals on the first real reset.
- Current refs: `contrastive/utils.py:305-333`, `env_utils.py:73-110`, `env_utils.py:152-188`, `env_utils.py:236-269`.

Why it matters:

- The actor loop can observe episode boundaries without resetting simulator state.
- Goals are kept stable across the long physical rollout.

Status: definitely non-episodic.

## 3. TimeStep Termination / Truncation
Original behavior:

- Raw Sawyer `step()` methods return `done = False`; true sequence boundaries come from `StepLimitWrapper`.
- Original refs: `main:env_utils.py:101-110`, `main:env_utils.py:169-182`, `main:env_utils.py:240-250`.

Non-episodic behavior:

- `FakeEpisodeBoundaryWrapper.step()` changes the timestep to `dm_env.StepType.LAST` after `steps_per_chunk`.
- It does not call the underlying reset when emitting this `LAST`.
- Current refs: `contrastive/utils.py:335-353`, `env_utils.py:112-121`, `env_utils.py:190-203`, `env_utils.py:271-281`.

Why it matters:

- `LAST` no longer means physical terminal state or true simulator truncation.
- It means "cut this continuous rollout into a replay trajectory here."

Status: definitely non-episodic.

## 4. Discount Handling
Original behavior:

- `ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn` copies `sample.data.discount[:-1]` into learner transitions.
- Future-state goal sampling uses `config.discount ** (k - t)`.
- Original refs: `main:contrastive/builder.py:119-152`.

Non-episodic behavior:

- Dataset code is effectively unchanged.
- Fake reset timesteps use `discount=1.0`.
- Fake `LAST` is created with `timestep._replace(step_type=LAST)` and does not explicitly change the underlying discount.
- Current refs: `contrastive/builder.py:123-156`, `contrastive/utils.py:318-323`, `contrastive/utils.py:340-343`.

Why it matters:

- The learner still consumes discount fields as before.
- Runtime semantics at fake `LAST` depend on what discount the wrapped timestep carried before `_replace`.

Status: possibly related; needs runtime verification.

## 5. Replay Sequence Boundaries
Original behavior:

- Replay uses `adders_reverb.EpisodeAdder`.
- `max_sequence_length=config.max_episode_steps + 1`.
- Original ref: `main:contrastive/builder.py:205-211`.

Non-episodic behavior:

- Replay still uses `EpisodeAdder` with the same max sequence contract.
- Boundaries are produced by fake `LAST` timesteps from `FakeEpisodeBoundaryWrapper`.
- Current refs: `contrastive/builder.py:209-217`, `contrastive/utils.py:340-353`.

Why it matters:

- Reverb still stores finite trajectories.
- The stored trajectories are chunks from one continuous rollout, not true independent episodes.

Status: definitely non-episodic.

## 6. Future-State Goal Sampling
Original behavior:

- `flatten_fn` samples a strictly future index within the stored trajectory:
  `is_future_mask[t, k] = t < k`.
- It weights candidates by `config.discount ** (k - t)`.
- Original refs: `main:contrastive/builder.py:119-140`.

Non-episodic behavior:

- The code is unchanged.
- Because replay trajectories are now artificial chunks, "future" means future within the chunk.
- Current refs: `contrastive/builder.py:123-145`.

Why it matters:

- SGCRL relabeling still cannot sample across a fake boundary.
- Long continuous rollouts are cut into finite relabeling windows.

Status: possibly related; unchanged code but changed meaning of the sequence.

## 7. Goal Relabeling Across Episode Boundaries
Original behavior:

- Future relabeling stays within one true episode stored by `EpisodeAdder`.
- Stored environment goals are discarded for learner targets; future states become goals.
- Original refs: `main:contrastive/builder.py:131-142`.

Non-episodic behavior:

- Future relabeling still stays within one stored sequence.
- A stored sequence is now an artificial chunk, so relabeling does not cross fake `LAST` boundaries even though the physical rollout continues.
- Current refs: `contrastive/builder.py:135-146`, `contrastive/utils.py:340-353`.

Why it matters:

- The branch removes physical episode boundaries, but not replay-window boundaries.
- This preserves original training code while limiting goal relabeling to chunk-local futures.

Status: definitely non-episodic in environment semantics; dataset behavior unchanged.

## 8. Dataset Iterator Behavior
Original behavior:

- `TrajectoryDataset.from_table_signature(...)` reads whole stored trajectories.
- It maps `flatten_fn`, transpose-shuffles, unbatches, then rebatches into `batch_size * num_sgd_steps_per_step`.
- Original refs: `main:contrastive/builder.py:164-203`.

Non-episodic behavior:

- Iterator logic is unchanged.
- It now receives fake episodes/chunks from continuous rollouts.
- Current refs: `contrastive/builder.py:168-207`.

Why it matters:

- No learner rewrite was needed.
- Any bugs from fake boundaries will enter through sampled trajectory contents, not through a new iterator algorithm.

Status: possibly related; same code, different input semantics.

## 9. Actor Loop Behavior
Original behavior:

- `DistributedLayout.actor()` creates an env via `self._environment_factory`, attaches `EpisodeAdder`, creates an actor, and returns `environment_loop.EnvironmentLoop`.
- Original/current ref: `contrastive/distributed_layout.py:233-257`.

Non-episodic behavior:

- Actor loop code is unchanged.
- The environment factory now returns an env wrapped by `FakeEpisodeBoundaryWrapper`.
- Current refs: `lp_contrastive.py:76-79`, `contrastive/utils.py:385-393`, `contrastive/distributed_layout.py:233-257`.

Why it matters:

- Acme still thinks it is running ordinary episodes.
- Non-episodic behavior is injected below the actor loop through wrapper semantics.

Status: definitely non-episodic at environment factory layer; actor loop unchanged.

## 10. Evaluator Behavior
Original behavior:

- Evaluator creates `environment_factory_fixed_goals` and runs `environment_loop.EnvironmentLoop` with observers.
- Original/current refs: `contrastive/distributed_layout.py:66-100`, `lp_contrastive.py:94-101`.

Non-episodic behavior:

- Evaluator loop code is unchanged.
- `environment_factory_fixed_goals` also calls `contrastive_utils.make_environment()`, so it likely gets the same fake-boundary wrapper.
- Current refs: `lp_contrastive.py:94-101`, `contrastive/utils.py:385-393`.

Why it matters:

- Evaluation may also be continuous/fake-episodic unless explicitly overridden elsewhere.
- Success/distance metrics may summarize artificial chunks, not independent real eval episodes.

Status: possibly related; needs runtime verification.

## 11. Episode Return / Success Logging
Original behavior:

- `SuccessObserver` records rewards until `observe_first()` starts the next episode, then stores whether any reward occurred.
- `DistanceObserver` stores per-episode initial/final/min distances and smooths history across episodes.
- Original/current refs: `contrastive/utils.py:37-64`, `contrastive/utils.py:67-120`.

Non-episodic behavior:

- Observer code is unchanged.
- Fake `FIRST` timesteps trigger `observe_first()` at chunk boundaries.
- Metrics therefore become chunk-level success/distance summaries.
- Current refs: `contrastive/utils.py:37-64`, `contrastive/utils.py:92-120`, `contrastive/utils.py:316-333`.

Why it matters:

- Logged "episode" metrics are no longer guaranteed to correspond to physical environment episodes.
- They are better read as chunk metrics under non-episodic training.

Status: definitely affected by non-episodic boundaries; observer code unchanged.

## Summary Table
| Assumption | Removed? | Where |
|---|---:|---|
| Episode length equals simulator reset horizon | Yes | `contrastive/utils.py::make_environment`, `FakeEpisodeBoundaryWrapper` |
| Reset means underlying env reset | Yes | `FakeEpisodeBoundaryWrapper.reset` |
| `LAST` means true terminal/truncated state | Yes | `FakeEpisodeBoundaryWrapper.step` |
| Replay sequence equals true episode | Yes | `EpisodeAdder` plus fake `LAST` |
| Future relabeling crosses continuous rollout boundaries | No | `ContrastiveBuilder.make_dataset_iterator` unchanged |
| Actor loop implementation changes | No | `DistributedLayout.actor` unchanged |
| Evaluator loop implementation changes | No | `default_evaluator_factory` unchanged |
| Success/distance metrics mean true episodes | Changed meaning | `SuccessObserver`, `DistanceObserver` unchanged, boundaries fake |

## Open Questions
1. What discount does Acme/Reverb store for fake `LAST` timesteps, since `FakeEpisodeBoundaryWrapper.step()` does not set discount explicitly?
2. Does evaluator behavior intentionally use fake non-episodic chunks, or should evaluation use true reset-based fixed-goal episodes?
3. Is `config.max_episode_steps = getattr(environment, '_step_limit') + 1` still correct after `make_environment()` sets `_step_limit = max_episode_steps`?
4. Does chunk-local future sampling lose important long-horizon positives from the continuous rollout?
5. Can MetaWorld internals still force resets or state changes despite `max_path_length = 10**9`?
