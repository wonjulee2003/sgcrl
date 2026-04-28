# Non-Episodic Branch Diff Map

## Scope
This note compares the current `non-episodic` branch against `main`, using `docs/onboarding/00_original_branch_master_summary.md` from `main` as the original SGCRL baseline.

Commands used:

```bash
git status --branch --short
git diff --stat main...HEAD
git diff --name-status main...HEAD
git diff --stat main..HEAD
git show main:docs/onboarding/00_original_branch_master_summary.md
```

Important comparison detail:

- `main...HEAD` compares changes introduced on `non-episodic` since the shared ancestor. This is the best view for code behavior.
- `main..HEAD` compares current `main` directly against current `non-episodic`. This also shows the onboarding docs committed only on `main` as absent from `non-episodic`.

Current branch at inspection time:

```text
non-episodic...origin/non-episodic
```

## Changed Files Summary
Using `git diff --stat main...HEAD`, the branch-level code diff is:

```text
.DS_Store                 deleted metadata
.gitignore                modified
contrastive/builder.py    modified
contrastive/networks.py   modified
contrastive/utils.py      modified
env_utils.py              modified
job.slurm                 added
lp_contrastive.py         modified
run_psi_sim.sh            added
run_tabular.sh            added
tabular_SGCRL.py          added
training_videos/.DS_Store deleted metadata
```

High-signal source changes:

- `contrastive/utils.py`: adds `FakeEpisodeBoundaryWrapper` and changes `make_environment()` to use continuous simulator rollouts with fake replay chunk boundaries.
- `env_utils.py`: changes Sawyer goal/reset behavior to support long physical episodes.
- `contrastive/builder.py`: keeps episode-based replay, but interprets stored episodes as chunks of one continuous rollout.
- `lp_contrastive.py`: adds non-episodic messaging, point-maze snapshot flags, and more fixed point-maze goals.
- `tabular_SGCRL.py`: new standalone tabular experiment with optional continuous-episode collection.

Notably unchanged in the code diff:

- `contrastive/learning.py`
- `contrastive/config.py`
- `contrastive/agents.py`
- `contrastive/distributed_layout.py`
- `point_env.py`

This strongly suggests the neural SGCRL non-episodic behavior is implemented at the environment/replay-boundary layer, not by changing the critic/actor losses.

## Added Files
| File | Purpose |
|---|---|
| `job.slurm` | Cluster launch script for `lp_contrastive.py`, currently targeting `sawyer_box`. |
| `run_psi_sim.sh` | Script for posterior/similarity diagnostics through `experiments.similarity_posterior_exp`; not part of the main SGCRL training path in the inspected baseline. |
| `run_tabular.sh` | Hyperparameter launcher for `tabular_SGCRL.py`; passes `--continuous-episode True`. |
| `tabular_SGCRL.py` | Standalone tabular SGCRL maze implementation with tabular `phi`/`psi`, replay, plotting, and optional continuous rollouts. |

Docs note:

- Current `main` also contains `docs/onboarding/*.md` files, including `00_original_branch_master_summary.md`.
- Those docs are not part of the original code snapshot on `non-episodic`. This file, `docs/onboarding/10_non_episodic_diff_map.md`, is being added only as onboarding analysis.

## Deleted Files
Using `main...HEAD`, only metadata files are deleted:

| File | Meaning |
|---|---|
| `.DS_Store` | Finder metadata removed. |
| `training_videos/.DS_Store` | Finder metadata removed. |

Using direct `main..HEAD`, the main-branch onboarding docs also appear deleted because they were added on `main` after the branch split. Treat those as documentation-branch skew, not non-episodic algorithm changes.

## Modified Files
| File | Main kind of change |
|---|---|
| `.gitignore` | Reworked ignored outputs, logs, notebook checkpoints, and `.DS_Store`. |
| `contrastive/builder.py` | Replay table buffer guard and comments clarifying chunked continuous episodes. |
| `contrastive/networks.py` | Adds unused `config=None` compatibility argument to `make_networks()`. |
| `contrastive/utils.py` | Core non-episodic wrapper logic and point-maze trajectory snapshot wrapper. |
| `env_utils.py` | Sawyer wrappers keep goals fixed after first reset and relax path length limits. |
| `lp_contrastive.py` | Runtime flags/messages and additional fixed point-maze goals. |

## Important Modified Files
### `contrastive/utils.py`
Structural changes:

- Adds `collections.deque`, `point_env`, and plotting-related support.
- Adds `_point_maze_wall_key(env_name)`.
- Adds `PointMazeTrajectorySnapshotWrapper`.
- Adds `FakeEpisodeBoundaryWrapper`.
- Changes `make_environment(env_name, start_index, end_index, seed, fixed_start_end=None, extra_dim=0)`.

Algorithmic changes:

- Original `make_environment()` wrapped the raw Gym env with:

```text
gym_wrapper.GymWrapper
  -> step_limit.StepLimitWrapper(step_limit=max_episode_steps)
  -> ObservationFilterWrapper
```

- Non-episodic branch wraps with:

```text
gym_wrapper.GymWrapper
  -> step_limit.StepLimitWrapper(step_limit=12_000_000)
  -> FakeEpisodeBoundaryWrapper(steps_per_chunk=max_episode_steps)
  -> ObservationFilterWrapper
  -> optional PointMazeTrajectorySnapshotWrapper
```

- `FakeEpisodeBoundaryWrapper.reset()` calls the real underlying `env.reset()` only on the first reset. Later resets return a synthetic `dm_env.TimeStep(step_type=FIRST, reward=0.0, discount=1.0, observation=self._last_observation)`.
- `FakeEpisodeBoundaryWrapper.step()` returns the real next timestep until `steps_per_chunk`; then it changes the timestep to `dm_env.StepType.LAST` without resetting the underlying simulator.

Why it relates to non-episodic training:

- Acme/Reverb still see finite trajectories, so `EpisodeAdder` and `TrajectoryDataset` can keep working.
- The physical/simulator state is not reset at each replay trajectory boundary.
- Replay episodes become fixed-length chunks of one long continuous rollout.

### `env_utils.py`
Structural changes:

- `SawyerBin.__init__`, `SawyerBox.__init__`, and `SawyerPeg.__init__` add `_goal_set_once`.
- These wrappers set `max_path_length` and `_max_path_length`, when present, to `10**9`.
- Their `reset()` methods still call `super().reset()`, but goal sampling/assignment now happens only if `_goal_set_once` is false.

Algorithmic changes:

- Original branch recomputed or reset task goals on every env reset.
- Non-episodic branch samples/sets the task goal once and reuses it across later resets.
- `step()` still returns `done = False` in the Sawyer wrappers.

Why it relates to non-episodic training:

- Prevents goal resampling at fake boundaries or long-running rollouts.
- Large MetaWorld path lengths avoid internal environment truncation during continuous physics.
- Keeps the target stable while the simulator is intended to continue over many replay chunks.

Open nuance:

- Because `FakeEpisodeBoundaryWrapper` avoids calling the underlying reset after the first real reset, `_goal_set_once` is most protective for any actual underlying reset that does occur. Runtime verification should confirm MetaWorld internals do not reset other state at path limits.

### `contrastive/builder.py`
Structural changes:

- `ContrastiveBuilder.make_replay_tables()` adds a minimum `error_buffer` guard:

```text
error_buffer = max(error_buffer, 2 * max(1.0, samples_per_insert))
```

- `ContrastiveBuilder.make_adder()` remains an `adders_reverb.EpisodeAdder`.

Algorithmic changes:

- The replay format is still episode/trajectory based.
- `ContrastiveBuilder.make_dataset_iterator()` is unchanged in the inspected diff: it still samples future goals within a stored trajectory using `t < k` and `discount ** (k - t)`.

Why it relates to non-episodic training:

- The branch does not replace episodic replay with independent transition replay.
- Instead, `FakeEpisodeBoundaryWrapper` makes continuous rollouts look like ordinary fixed-length episodes to `EpisodeAdder`.
- This preserves the original SGCRL learner input contract while changing what a replay "episode" means.

### `lp_contrastive.py`
Structural changes:

- Adds flags `--maze_traj_snapshot_every` and `--maze_traj_snapshot_history`.
- Adds more point-maze entries in `fixed_goal_dict`: `point_Impossible`, `point_Maze11x11`, and `point_Wall11x11`.
- Prints a runtime message that environments use `FakeEpisodeBoundaryWrapper`.
- Sets `SGCRL_MAZE_TRAJ_*` environment variables before launch.

Algorithmic changes:

- Core algorithm dispatch is unchanged: `contrastive_cpc`, `c_learning`, and `nce+c_learning` set the same config flags as before.
- `get_program()` still sets `config.obs_dim`, `config.max_episode_steps`, builds `network_factory`, and constructs `DistributedContrastive`.

Why it relates to non-episodic training:

- Makes non-episodic behavior explicit at launch.
- Enables point-maze visual diagnostics for continuous movement.
- Uses the modified `contrastive.utils.make_environment()`, so all normal launches receive fake-boundary continuous environments.

### `contrastive/networks.py`
Structural changes:

- `make_networks()` now accepts `config=None`.
- The argument is explicitly unused and kept for compatibility.

Algorithmic changes:

- No observed change to policy, critic, `phi`, `psi`, `[B, B]` logits, or actor/critic losses.

Why it relates to non-episodic training:

- Probably not directly related. It may support callers that pass config into the network factory.

### `.gitignore`
Structural changes:

- Removes many broad artifact patterns from `main` and adds branch-specific output paths such as `joblog/`, `archived/`, `eval_plots/`, `runs_density/`, and `runs_final/`.

Algorithmic changes:

- None.

Why it relates to non-episodic training:

- Mostly operational/logging cleanup, not algorithmic.

## Behavior Table
| File | Original behavior | Non-episodic behavior | Why it matters |
|---|---|---|---|
| `contrastive/utils.py::make_environment` | Uses `StepLimitWrapper(step_limit=max_episode_steps)` to create normal finite episodes. | Uses `StepLimitWrapper(step_limit=12_000_000)` plus `FakeEpisodeBoundaryWrapper(steps_per_chunk=max_episode_steps)`. | Decouples simulator resets from replay trajectory boundaries. |
| `contrastive/utils.py::FakeEpisodeBoundaryWrapper` | Did not exist. | Emits fake `LAST` every chunk and fake `FIRST` on next reset while preserving underlying simulator state. | Core mechanism for non-episodic rollouts while keeping Acme/Reverb episode APIs. |
| `contrastive/builder.py::make_adder` | `EpisodeAdder(max_sequence_length=config.max_episode_steps + 1)` stores finite episodes. | Same API, but input "episodes" are fake chunks of a continuous rollout. | Avoids rewriting learner/replay code. |
| `contrastive/builder.py::make_dataset_iterator` | Samples future goals within each stored trajectory. | Same logic. Stored trajectories are continuous chunks, not true environment episodes. | Future-state relabeling now happens within artificial chunks. |
| `env_utils.py::SawyerBin.reset` | Recomputes goal on reset. | Sets goal only once via `_goal_set_once`. | Avoids goal changes across long-running training rollouts. |
| `env_utils.py::SawyerBox.reset` | Recomputes goal/quat on reset. | Sets goal/quat only once via `_goal_set_once`. | Keeps Sawyer box target stable across continuous chunks. |
| `env_utils.py::SawyerPeg.reset` | Recomputes goal on reset. | Sets goal only once via `_goal_set_once`. | Keeps Sawyer peg target stable across continuous chunks. |
| `env_utils.py::Sawyer*.__init__` | Uses default MetaWorld path length behavior. | Sets path length fields to `10**9`. | Reduces chance of MetaWorld internal truncation. |
| `lp_contrastive.py::main` | Launches normal SGCRL without explicit continuous-episode message. | Prints non-episodic physics message and configures maze trajectory snapshots. | Makes runtime mode visible and adds diagnostics. |
| `contrastive/networks.py::make_networks` | Does not accept `config`. | Accepts unused `config=None`. | Compatibility only; no apparent algorithm change. |
| `contrastive/learning.py` | Actor/critic losses and gradient update path. | Unchanged in `main...HEAD`. | Non-episodic behavior is not implemented in the learner losses. |

## Likely Non-Episodic Design Changes
1. Continuous simulator state:
   `FakeEpisodeBoundaryWrapper` tries to keep one physical rollout alive after the first real reset.

2. Fake replay boundaries:
   The wrapper emits synthetic `LAST` and `FIRST` timesteps so `EpisodeAdder` can still write finite trajectories.

3. Fixed chunk length:
   Chunks are cut every original `max_episode_steps`, not when the real simulator terminates.

4. Original learner compatibility:
   `ContrastiveLearner`, `ContrastiveBuilder.make_dataset_iterator()`, and the `[state, goal]` observation contract are preserved.

5. Goal stability:
   Sawyer task goals are set once and retained, matching the idea of a long-running task instead of independent per-episode tasks.

6. Long path limits:
   MetaWorld path lengths and Acme step limits are inflated to prevent accidental real episode termination.

7. Diagnostics for continuous movement:
   Point-maze trajectory snapshots visualize recent positions and current goal over time.

## Changes That Seem Unrelated To Non-Episodic Behavior
- `.gitignore` cleanup and branch-specific artifact paths.
- `.DS_Store` deletions.
- `contrastive/networks.py::make_networks(config=None)` compatibility parameter.
- `job.slurm` cluster details, CUDA/JAX memory settings, and user-specific paths.
- `run_psi_sim.sh` posterior/similarity plotting script.
- Much of `tabular_SGCRL.py` plotting, logging, EMA metrics, and visualization machinery.
- `lp_contrastive.py` checkpoint interval change from 5 to 15 minutes.

## Added Tabular Experiment
`tabular_SGCRL.py` is not part of the distributed neural SGCRL path described in `00_original_branch_master_summary.md`, but it mirrors the non-episodic idea in a simpler setting.

Relevant pieces:

- `parse_args()` adds `--continuous-episode`.
- `collect_episode()` uses `current_env_state` as the next segment start when `continuous_episode` is true.
- `run_tabular.sh` launches `tabular_SGCRL.py` with `--continuous-episode True`.

Interpretation:

- In episodic mode, each collected segment starts from `START_STATE`.
- In continuous mode, each segment starts from the previous segment's final state, but the replay buffer still stores fixed-length trajectories.
- This is the tabular analogue of `FakeEpisodeBoundaryWrapper`.

## Current Main vs Current Non-Episodic Docs Difference
Direct `git diff --stat main..HEAD` includes the current `main` onboarding docs as deleted from `non-episodic`:

```text
docs/onboarding/00_codebase_map_short.md
docs/onboarding/00_original_branch_master_summary.md
docs/onboarding/01_training_step_trace_short.md
docs/onboarding/02_tensor_shape_ledger_short.md
docs/onboarding/03_goal_flow_short.md
docs/onboarding/04_algorithm_variants_short.md
docs/onboarding/05_networks_and_representations_short.md
docs/onboarding/06_distributed_infrastructure.md
docs/onboarding/07_agent_layout_builder_stack.md
```

These are documentation differences from `main`, not source-code changes that make the branch non-episodic.

## Open Questions / Needs Runtime Verification
1. Does `FakeEpisodeBoundaryWrapper` fully prevent underlying simulator resets in actor and evaluator loops, or can external wrappers still force a real reset?
2. Does `StepLimitWrapper(step_limit=12_000_000)` ever produce a true terminal timestep during long jobs?
3. Does `env._step_limit = max_episode_steps` on the wrapped env reliably propagate to `lp_contrastive.py::get_program`, where `config.max_episode_steps = getattr(environment, '_step_limit') + 1`?
4. Is the `+ 1` in `config.max_episode_steps = getattr(environment, '_step_limit') + 1` still correct when `FakeEpisodeBoundaryWrapper` chunks at `max_episode_steps`?
5. Are fake `LAST` timesteps assigned the discount semantics expected by `ContrastiveBuilder.make_dataset_iterator()` and TD/C-learning?
6. Since `make_dataset_iterator()` still samples future goals only within a chunk, does chunking lose useful long-horizon future states from the same continuous rollout?
7. Should evaluation also be non-episodic, or should evaluator resets remain meaningful fixed-goal trials?
8. Do Sawyer wrappers need additional state preservation beyond goal stability, given their `reset()` methods still call `super().reset()` on the first real reset?
9. Are point-maze trajectory snapshots too expensive or disk-heavy for long actor jobs with many actors?
10. Is `run_tabular.sh` syntactically correct around the line-continuation after `--continuous-episode True`, or does the blank line break the `nohup python` command before redirection?
11. Does `contrastive/networks.py::make_networks(config=None)` correspond to an external caller not present in this branch?
12. Should the branch eventually copy the `main` onboarding docs so future comparison notes live on both branches?
