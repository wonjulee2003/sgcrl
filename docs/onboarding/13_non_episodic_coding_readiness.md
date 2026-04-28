# Non-Episodic Coding Readiness

## Core Mechanism
Plain English:

The branch keeps the original SGCRL learner almost unchanged. Instead of making the learner understand infinite/non-episodic data, it makes one long simulator rollout look like many ordinary finite episodes.

Mechanism:

```text
real long env rollout
  -> FakeEpisodeBoundaryWrapper inserts fake LAST every chunk
  -> next reset is fake FIRST, not real env.reset()
  -> EpisodeAdder stores each chunk as a trajectory
  -> learner samples chunks and runs original SGCRL losses
```

So the branch is "non-episodic" at the environment/replay-boundary layer, not at the loss-function layer.

## 5 Most Important Changed Files
1. `contrastive/utils.py`
   Core fake-boundary wrapper and modified environment construction.

2. `env_utils.py`
   Sawyer envs keep goals fixed after first setup and relax MetaWorld path limits.

3. `contrastive/builder.py`
   Replay still uses `EpisodeAdder`; stored episodes now mean continuous-rollout chunks.

4. `lp_contrastive.py`
   Builds env factories that now return fake-boundary environments; adds launch-time diagnostics.

5. `tabular_SGCRL.py`
   Separate tabular analogue of continuous chunked rollouts; useful for conceptual comparison, not the neural distributed path.

## 10 Most Important Changed Functions / Classes
1. `contrastive.utils.FakeEpisodeBoundaryWrapper`
2. `contrastive.utils.FakeEpisodeBoundaryWrapper.step`
3. `contrastive.utils.FakeEpisodeBoundaryWrapper.reset`
4. `contrastive.utils.make_environment`
5. `contrastive.utils.PointMazeTrajectorySnapshotWrapper`
6. `env_utils.SawyerBin.reset`
7. `env_utils.SawyerBox.reset`
8. `env_utils.SawyerPeg.reset`
9. `contrastive.builder.ContrastiveBuilder.make_replay_tables`
10. `lp_contrastive.get_program`

Also watch:

- `contrastive.builder.ContrastiveBuilder.make_adder`
- `contrastive.builder.ContrastiveBuilder.make_dataset_iterator`
- `contrastive.utils.SuccessObserver`
- `contrastive.utils.DistanceObserver`

These are not heavily changed, but their input semantics changed.

## Behavior Not To Accidentally Break
- Fake `LAST` must end replay chunks without resetting the underlying simulator.
- Fake `FIRST` must reuse `self._last_observation`.
- The first reset must still be a real reset.
- `EpisodeAdder` must still receive finite trajectories.
- `make_dataset_iterator()` must still receive full sequences and use `[:-1]` / `[1:]`.
- Observation format must remain `[state, goal]`.
- `config.obs_dim` must still split state and goal correctly.
- Future relabeling must still sample only valid future indices within the stored sequence.
- `critic_loss()` must still see matched diagonal positives and in-batch off-diagonal negatives.
- `actor_loss()` must still maximize `diag(Q(s, a, g))`.
- Target critic update must remain independent of fake-boundary logic.
- Sawyer fixed-goal behavior must not silently resample goals mid-rollout.
- Evaluator behavior must be understood before changing wrappers, because it likely also gets fake boundaries.

## Original Assumptions That No Longer Hold
| Original assumption | Non-episodic branch reality |
|---|---|
| One replay sequence equals one real episode. | One replay sequence is a chunk of a longer rollout. |
| `env.reset()` means simulator reset. | Wrapper resets after the first are fake. |
| `LAST` means true terminal/truncation. | `LAST` can mean "cut replay here." |
| Episode metrics are true episode metrics. | They may be chunk metrics. |
| Future goals are future states in one true episode. | They are future states in one artificial chunk. |
| Step limit equals task horizon. | Original horizon becomes chunk length; wrapper limit is huge. |
| Goal can be resampled on every reset. | Sawyer goals are intended to stay fixed after first setup. |
| Evaluator is definitely episodic. | Needs verification; it uses the same environment factory path. |

## Smoke Tests Before Editing
Run these before making behavior changes:

```bash
git status --branch --short
git diff --name-status main...HEAD
python -m py_compile contrastive/utils.py contrastive/builder.py env_utils.py lp_contrastive.py
```

Inspect branch-specific assumptions:

```bash
rg -n "FakeEpisodeBoundaryWrapper|continuous-episode|_goal_set_once|12_000_000|StepType.LAST|StepType.FIRST" contrastive env_utils.py lp_contrastive.py
```

Minimal wrapper sanity test to write/run before deeper edits:

```text
construct env via contrastive.utils.make_environment(...)
call reset once -> expect REAL reset log
step until chunk boundary -> expect fake LAST
call reset again -> expect fake FIRST and no underlying reset log
verify observation shape remains [state, goal]
```

If dependencies are available, run the smallest local Launchpad job you can tolerate:

```bash
python -u lp_contrastive.py --env point_Spiral11x11 --alg contrastive_cpc --num_steps 1000 --maze_traj_snapshot_every 0
```

Expected logs to look for:

```text
[sgcrl continuous-episode ...] REAL reset
[sgcrl continuous-episode ...] FAKE LAST
[sgcrl continuous-episode ...] FAKE reset
```

## Where To Add Logging
Best places:

- `contrastive.utils.FakeEpisodeBoundaryWrapper.__init__`
  Log chunk length, env name, process id, and wrapper stack.

- `contrastive.utils.FakeEpisodeBoundaryWrapper.step`
  Log fake `LAST` count, step count, reward, discount, and observation summary.

- `contrastive.utils.FakeEpisodeBoundaryWrapper.reset`
  Log real vs fake reset count and whether `self._last_observation` exists.

- `contrastive.builder.ContrastiveBuilder.make_adder`
  Log `config.max_episode_steps` and `max_sequence_length`.

- `contrastive.builder.ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn`
  Harder because it is TensorFlow graph code; use sparingly. If needed, log `seq_len`, sampled `goal_index`, and discount summaries.

- `lp_contrastive.get_program`
  Log `environment._step_limit`, `config.max_episode_steps`, `obs_dim`, and env name.

- `contrastive.utils.SuccessObserver.observe_first`
  If interpreting results, log that "episode" means fake chunk under this branch.

## Suggested First Small Coding Task
Add an opt-in debug mode for `FakeEpisodeBoundaryWrapper`.

Goal:

- Count real resets, fake resets, and fake `LAST`s.
- Print or expose these counts every N chunks.
- Include discount at fake `LAST`.

Why this first:

- It verifies the central non-episodic mechanism without touching learner math.
- It answers the biggest runtime question: what exactly does Reverb see at chunk boundaries?

## Suggested Second Coding Task
Add a tiny wrapper/unit smoke test for fake boundaries.

Goal:

- Use a minimal dummy `dm_env.Environment`.
- Wrap it with `FakeEpisodeBoundaryWrapper(steps_per_chunk=3)`.
- Assert first reset is real, step 3 returns `LAST`, next reset returns `FIRST` with previous observation, and the dummy env reset count stays `1`.

Why this second:

- It protects the branch from accidental regressions in the non-episodic contract.
- It avoids needing MuJoCo/MetaWorld just to test the core wrapper.

## Open Questions For Mentor / Lab
1. Should evaluator runs be fake-boundary non-episodic, or true reset-based episodes?
2. Should future-goal relabeling ever cross fake chunk boundaries?
3. What discount should fake `LAST` carry: inherited discount, `0.0`, or `1.0`?
4. Is `config.max_episode_steps = getattr(environment, '_step_limit') + 1` intentional with `EpisodeAdder(max_sequence_length=config.max_episode_steps + 1)`?
5. Are Sawyer goals supposed to be fixed for all experiments, including `--sample_goals=True`?
6. Should point-maze and Sawyer environments have the same non-episodic wrapper semantics?
7. Is the tabular continuous-episode implementation intended as the reference behavior for neural SGCRL?
8. Should success/distance metrics be renamed or duplicated as `chunk_success`, `chunk_final_dist`, etc.?
9. Should fake-boundary logs be always on, env-var gated, or logger-based?
10. Is the actor update using `state.q_params` instead of the freshly updated `q_params` intentional?
