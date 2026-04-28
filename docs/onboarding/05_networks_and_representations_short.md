# Networks And Representations Short

## Big Picture

Networks are built in `contrastive.networks.make_networks` and returned as `contrastive.networks.ContrastiveNetworks`.

Main pieces:

- `policy_network`: maps `[state, goal]` to a bounded action distribution.
- `q_network`: maps `[state, goal]` plus `action` to contrastive logits.
- `sample`, `sample_eval`, `log_prob`: helpers around the action distribution.

## Key Dimensions

| Symbol | Meaning |
|---|---|
| `B` | critic batch size |
| `M` | actor-loss batch size |
| `D` | state dimension / `obs_dim` |
| `G` | goal dimension |
| `A` | action dimension |
| `R` | representation dimension, default `64` |

## Policy Network

File/function:

```text
contrastive.networks.make_networks.<locals>._actor_fn
```

Input:

```text
obs = [state, goal]   [M, D + G]
```

Architecture:

```text
hk.nets.MLP(hidden_layer_sizes)
-> distributional.NormalTanhDistribution(A)
```

Default hidden layers:

```text
(256, 256)
```

Output:

```text
pi_theta(a | s, g)
tanh-normal distribution over actions [M, A]
```

Action distribution classes:

```text
distributional.NormalTanhDistribution
distributional.TanhTransformedDistribution
```

The tanh transform keeps actions in `[-1, 1]`.

## Policy Sampling

File/function: `contrastive.networks.apply_policy_and_sample`

| Mode | Action choice |
|---|---|
| `eval_mode=False` | `sample(policy_distribution)` |
| `eval_mode=True` | `mode(policy_distribution)` |

## Critic Network

File/function:

```text
contrastive.networks.make_networks.<locals>._critic_fn
```

Inputs:

```text
obs       [B, D + G]
action    [B, A]
```

Output:

```text
logits   [B, B]
sa_repr  [B, R]
g_repr   [B, R]
```

If `twin_q=True`, intended logits shape is `[B, B, 2]`.

## Representations

File/function:

```text
contrastive.networks.make_networks.<locals>._repr_fn
```

Split observation:

```text
state = obs[:, :obs_dim]   [B, D]
goal  = obs[:, obs_dim:]   [B, G]
```

State-action representation:

```text
sa_repr = sa_encoder(concat([state, action]))
concat([state, action])  [B, D + A]
sa_repr                  [B, R]
```

Goal representation:

```text
g_repr = g_encoder(goal)
goal    [B, G]
g_repr  [B, R]
```

Mathematically:

```text
sa_repr_i = phi(s_i, a_i)
g_repr_j  = psi(g_j)
```

## Combining Representations

File/function:

```text
contrastive.networks.make_networks.<locals>._combine_repr
```

Formula:

```text
logits[i, j] = dot(sa_repr[i], g_repr[j])
logits shape = [B, B]
```

Meaning:

```text
diagonal logits[i, i]     = matched state-action and goal, positive
off-diagonal logits[i, j] = mismatched goal, negative
```

This `[B, B]` matrix is the core contrastive object.

## How Learner Uses Logits

File/function:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.critic_loss
```

Branches:

- CPC: `optax.softmax_cross_entropy(logits, labels=eye(B))`
- NCE-style latent branch: `optax.sigmoid_binary_cross_entropy(logits, labels=eye(B))`
- TD/C-learning: uses `sigmoid(target_q_logits)`, `target_q_params`, and `stop_gradient(next_v)`

Actor loss uses the critic in:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss
```

It samples actions from the policy and maximizes the diagonal critic score:

```text
actor_loss = mean(-diag(q_action))
```

## Initialization And Optimization

Haiku setup in `contrastive.networks.make_networks`:

```text
policy = hk.without_apply_rng(hk.transform(_actor_fn))
critic = hk.without_apply_rng(hk.transform(_critic_fn))
```

Returned as Acme wrappers:

```text
networks_lib.FeedForwardNetwork(init, apply)
```

Optimizers are created in:

```text
contrastive.builder.ContrastiveBuilder.make_learner
```

Updates happen in:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.update_step
```

## Open Questions

1. `twin_q=True` branch references `product`, but first critic matrix is named `critic_val`.
2. `repr_norm_temp` exists but is not passed by `lp_contrastive.py::get_program`.
3. `actor_loss` receives `sa_repr` and `sf_repr` but does not use them.

