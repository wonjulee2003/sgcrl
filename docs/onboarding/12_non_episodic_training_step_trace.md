# Non-Episodic Training Step Trace

## Scope
Trace one learner update on current branch `non-episodic`, compared against `main`.

This is not a full original-code walkthrough. The key point is that `contrastive/learning.py` is unchanged in `main...HEAD`; the learner receives the same tensor shapes and runs the same losses. What changes is the meaning of the replay trajectory: it is now a chunk from a continuous rollout, not necessarily a true environment episode.

## Before The Learner: Where The Difference Enters
Original behavior:

- `main:contrastive/utils.py::make_environment` used `StepLimitWrapper(step_limit=max_episode_steps)`.
- `EpisodeAdder` received true finite episodes/truncations from Acme's environment loop.

Non-episodic behavior:

- `contrastive.utils.make_environment()` uses `StepLimitWrapper(step_limit=12_000_000)`, then wraps with `FakeEpisodeBoundaryWrapper(steps_per_chunk=max_episode_steps)`.
- `FakeEpisodeBoundaryWrapper.step()` emits fake `LAST` timesteps at chunk boundaries.
- `FakeEpisodeBoundaryWrapper.reset()` later emits fake `FIRST` timesteps without underlying `env.reset()`.
- Refs: `contrastive/utils.py:271-353`, `contrastive/utils.py:356-393`, `lp_contrastive.py:76-101`.

Mathematical meaning:

```text
Original replay sequence: tau = (x_0, ..., x_T) from one environment episode.
Non-episodic sequence:    tau_c = (x_c, ..., x_{c+T}) from a longer continuous rollout.
```

Central to non-episodic SGCRL: yes.

## 1. Replay Sample Structure
Original behavior:

- `ContrastiveBuilder.make_adder()` uses `adders_reverb.EpisodeAdder(max_sequence_length=config.max_episode_steps + 1)`.
- Reverb stores trajectory samples that correspond to finite episodes.
- Original ref: `main:contrastive/builder.py:205-211`.

Non-episodic behavior:

- `ContrastiveBuilder.make_adder()` still uses `EpisodeAdder`.
- The sequence boundary is now produced by fake `LAST`, so the sample is a chunk from a continuous rollout.
- Current refs: `contrastive/builder.py:209-217`, `contrastive/utils.py:340-353`.

Mathematical meaning:

```text
sample.data.observation = [o_c, o_{c+1}, ..., o_{c+T}]
```

where `c` is a chunk start, not necessarily a physical reset.

Central: yes, because this is how non-episodic data is made compatible with episodic replay.

## 2. Transition Construction
Original behavior:

- `flatten_fn(sample)` converts one stored trajectory into transitions:
  `observation[:-1]`, `action[:-1]`, `reward[:-1]`, `discount[:-1]`, `next_observation[1:]`.
- Original ref: `main:contrastive/builder.py:119-152`.

Non-episodic behavior:

- Code is unchanged.
- It now constructs transitions from an artificial continuous-rollout chunk.
- Current ref: `contrastive/builder.py:123-156`.

Mathematical meaning:

```text
transition_t = (s_t, a_t, r_t, d_t, s_{t+1})
```

but `t` ranges over a chunk-local index.

Central: possibly. The code is not changed, but the same construction now operates on fake episodes.

## 3. Goal Sampling
Original behavior:

- `flatten_fn` samples a strictly future index `k` in the same stored episode:
  `is_future_mask[t, k] = t < k`.
- Probability is proportional to `config.discount ** (k - t)`.
- It uses future state coordinates as the training goal.
- Original ref: `main:contrastive/builder.py:120-145`.

Non-episodic behavior:

- Goal sampling code is unchanged.
- Future goals are sampled only within the artificial chunk, not across fake chunk boundaries.
- Current refs: `contrastive/builder.py:124-145`, `contrastive/utils.py:340-353`.

Mathematical meaning:

```text
P(k | t) ∝ 1[t < k] * gamma^(k - t)
g_t = h(s_k)
```

where `k` is future-within-chunk.

Central: yes in effect, because goal relabeling now uses chunk-local future states from a continuous rollout.

## 4. Discount / Termination Handling
Original behavior:

- `flatten_fn` copies `sample.data.discount[:-1]` into `Transition.discount`.
- `config.discount` also weights future-goal sampling and TD/C-learning terms.
- Original refs: `main:contrastive/builder.py:123-152`, `main:contrastive/learning.py:172-177`.

Non-episodic behavior:

- `Transition.discount` construction is unchanged.
- Fake reset uses `discount=1.0`.
- Fake `LAST` changes only `step_type`; it does not explicitly overwrite discount.
- Current refs: `contrastive/builder.py:127-156`, `contrastive/utils.py:318-323`, `contrastive/utils.py:340-343`, `contrastive/learning.py:172-177`.

Mathematical meaning:

- Learner-side future sampling still uses `gamma`.
- Termination mostly affects where trajectories are cut before sampling.
- In inspected learner losses, `transitions.discount` is carried but not visibly used by `critic_loss`.

Central: possibly. Boundary semantics are central; exact discount semantics need runtime verification.

## 5. Critic Loss
Original behavior:

- `ContrastiveLearner.step()` pulls a replay batch, builds `types.Transition`, and calls `_update_step`.
- `critic_loss()` forms logits with `networks.q_network.apply(q_params, transitions.observation, transitions.action)`.
- CPC branch uses softmax CE; non-CPC MC branch uses sigmoid BCE; TD branch uses bootstrapped sigmoid/BCE terms.
- Original refs: `main:contrastive/learning.py:101-213`, `main:contrastive/learning.py:387-391`.

Non-episodic behavior:

- Critic loss code is unchanged.
- The batch distribution changes because transitions come from continuous-rollout chunks.
- Current refs: `contrastive/learning.py:101-213`, `contrastive/learning.py:387-391`.

Mathematical meaning:

```text
logits[i, j] = phi(s_i, a_i)^T psi(g_j)
```

The diagonal still means matched relabeled pair. The off-diagonals are in-batch negatives. The difference is the source of `g_i`: future states inside a fake chunk.

Central: no code change, but central to how non-episodic data affects learning.

## 6. Actor Loss
Original behavior:

- `actor_loss()` samples actions from `pi(. | s, g)`, evaluates critic scores, and minimizes `-diag(q_action)`.
- Original ref: `main:contrastive/learning.py:215-261`.

Non-episodic behavior:

- Actor loss code is unchanged.
- The states/goals seen by the actor during updates come from fake-chunk relabeled transitions.
- Current ref: `contrastive/learning.py:215-261`.

Mathematical meaning:

```text
min_pi E[-Q(s, a_pi, g)] = max_pi E[phi(s, a_pi)^T psi(g)]
```

with `g` sampled from future states in a continuous-rollout chunk.

Central: no code change; data semantics changed.

## 7. Target Network Updates
Original behavior:

- `update_step()` applies critic gradients to `q_params`.
- Then it soft-updates `target_q_params = (1 - tau) target_q_params + tau q_params`.
- Original ref: `main:contrastive/learning.py:282-292`.

Non-episodic behavior:

- Target update code is unchanged.
- It tracks the critic trained on non-episodic chunk data.
- Current refs: `contrastive/learning.py:282-292`, `contrastive/learning.py:312-319`.

Mathematical meaning:

```text
theta_target <- (1 - tau) theta_target + tau theta_q
```

Central: no. It is standard learner machinery, unchanged.

## 8. Metrics / Logging
Original behavior:

- Learner logs critic/actor metrics plus `steps_per_second`.
- Episode observers log success/distance per environment episode.
- Original refs: `main:contrastive/learning.py:305-310`, `main:contrastive/learning.py:393-407`, `main:contrastive/utils.py:37-120`.

Non-episodic behavior:

- Learner logging code is unchanged.
- Observer episode metrics are now likely chunk metrics, because fake `FIRST`/`LAST` timesteps define observed episodes.
- Current refs: `contrastive/learning.py:305-310`, `contrastive/learning.py:393-407`, `contrastive/utils.py:37-120`, `contrastive/utils.py:316-353`.

Mathematical meaning:

- `critic_loss`, `actor_loss`, accuracies, and logit stats mean the same.
- `success`, `init_dist`, `final_dist`, and `min_dist` may summarize fake chunks, not true simulator episodes.

Central: yes for interpretation, not for the gradient update itself.

## One Non-Episodic Learner Step In 10 Lines
1. Actor env runs a long physical rollout through `FakeEpisodeBoundaryWrapper`.
2. Every `steps_per_chunk`, wrapper emits fake `LAST`; next reset is fake `FIRST`.
3. `EpisodeAdder` writes that chunk as if it were an episode.
4. Reverb returns a trajectory chunk to `make_dataset_iterator()`.
5. `flatten_fn` samples future states within the chunk as goals.
6. It builds `Transition([s_t, g_t], a_t, r_t, d_t, [s_{t+1}, g_t])`.
7. `ContrastiveLearner.step()` gets a batched `Transition`.
8. `critic_loss()` updates `phi`/`psi` using the same CPC/NCE/TD objective as `main`.
9. `actor_loss()` updates the policy to maximize diagonal critic score for relabeled goals.
10. Target critic and metrics update exactly as in `main`.

## 5 Most Important Changed Functions
1. `contrastive.utils.make_environment`
2. `contrastive.utils.FakeEpisodeBoundaryWrapper.step`
3. `contrastive.utils.FakeEpisodeBoundaryWrapper.reset`
4. `contrastive.builder.ContrastiveBuilder.make_adder`
5. `contrastive.builder.ContrastiveBuilder.make_dataset_iterator`

Note: items 4 and 5 are only lightly changed or unchanged in code, but they are where fake chunks become learner batches.

## Open Questions
1. What discount is stored at fake `LAST` boundaries when `timestep._replace(step_type=LAST)` does not set `discount`?
2. Should future-goal sampling cross fake chunk boundaries in a truly non-episodic formulation?
3. Is `config.max_episode_steps = getattr(environment, '_step_limit') + 1` still consistent with `EpisodeAdder(max_sequence_length=config.max_episode_steps + 1)`?
4. Should evaluator metrics be true reset-based episodes or fake continuous chunks?
5. Does the actor update intentionally use `state.q_params` rather than the just-updated local `q_params`?
