# Tensor Shape Ledger Short

## Dimension Cheat Sheet

| Name | Meaning | Default / Example |
|---|---|---|
| `D` / `obs_dim` | state dimension | env-specific |
| `G` | goal dimension | usually `D` |
| `D + G` | full observation `[state, goal]` | env-specific |
| `A` | action dimension | from `spec.actions.shape` |
| `B` | learner batch size | `256` |
| `N` | SGD updates per learner step | `64` |
| `B_total` | replay batch per public step, `B * N` | `16384` |
| `R` | representation dimension | `64` |
| `M` | actor-loss batch after goal mixing | default `2B = 512` |

## Environment Observation Dimensions

| Env | `D` | `G` default | Observation `D + G` |
|---|---:|---:|---:|
| `point_Spiral11x11` | `2` | `2` | `4` |
| `sawyer_bin` | `7` | `7` | `14` |
| `sawyer_box` | `11` | `11` | `22` |
| `sawyer_peg` | `7` | `7` | `14` |

References:

- `env_utils.load`
- `env_utils.SawyerBin.observation_space`
- `env_utils.SawyerBox.observation_space`
- `env_utils.SawyerPeg.observation_space`
- `point_env.PointEnv.observation_space`

## The Most Important Split

The observation is flat:

```text
obs = [state, goal]
obs shape = [B, D + G]
```

The code splits it with `obs_dim`:

```text
state = obs[:, :D]      [B, D]
goal  = obs[:, D:]      [B, G]
```

References:

- `contrastive.learning.ContrastiveLearner.__init__.<locals>.critic_loss`
- `contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss`
- `contrastive.networks.make_networks.<locals>._repr_fn`

## Replay Shapes

File/function:

```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator
```

Before relabeling:

```text
sample.data.observation  [S, D + G]
sample.data.action       [S, A]
sample.data.reward       [S]
sample.data.discount     [S]
```

After future-goal relabeling:

```text
transition.observation       [T, D + G]
transition.action            [T, A]
transition.reward            [T]
transition.discount          [T]
transition.next_observation  [T, D + G]
```

Final public learner batch:

```text
observation       [B_total, D + G]
action            [B_total, A]
reward            [B_total]
discount          [B_total]
next_observation  [B_total, D + G]
```

With defaults:

```text
B_total = 256 * 64 = 16384
```

## Critic Shapes

Inputs:

```text
obs       [B, D + G]
action    [B, A]
```

Representations:

```text
concat([state, action])  [B, D + A]
sa_repr                  [B, R]
g_repr                   [B, R]
```

Logits:

```text
logits[i, j] = dot(sa_repr[i], g_repr[j])
logits shape = [B, B]
```

Twin critic:

```text
logits shape = [B, B, 2]
```

## Why `[B, B]`?

For each row `i`, the critic compares one state-action pair against every goal in the batch:

```text
row i = phi(s_i, a_i)
column j = psi(g_j)
```

So:

```text
logits[i, i] = matched goal, positive
logits[i, j] = other goal, negative
```

The label matrix is:

```text
I = jnp.eye(B)     [B, B]
```

## Actor Shapes

Default `config.random_goals=0.5` doubles the actor batch:

```text
new_state  [2B, D]
new_goal   [2B, G]
new_obs    [2B, D + G]
```

Policy:

```text
dist_params  batch [M], event [A]
action       [M, A]
log_prob     [M]
```

Critic for actor loss:

```text
q_action      [M, M]
diag(q_action) [M]
actor_loss    scalar
```

## Open Questions

1. Sawyer action dimension `A` is inherited from MetaWorld, not written directly in this repo.
2. Verify `twin_q=True` output shape because of the `product` reference in `_critic_fn`.
3. Verify exact splitting behavior of `acme.jax.utils.process_multiple_batches`.

