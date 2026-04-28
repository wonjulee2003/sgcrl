# Training Step Trace Short

## Big Picture

One public learner step starts at `contrastive.learning.ContrastiveLearner.step`. It gets a replay batch, converts it into `acme.types.Transition`, runs a multi-batch JAX update, updates critic/policy/target critic, and logs metrics.

## Main Call Path

```text
contrastive.learning.ContrastiveLearner.step
  -> sample = next(self._iterator)
  -> transitions = types.Transition(*sample.data)
  -> self._state, metrics = self._update_step(self._state, transitions)
  -> self._counter.increment(...)
  -> self._logger.write(...)
```

The iterator is created in:

```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator
```

The learner is created in:

```text
contrastive.builder.ContrastiveBuilder.make_learner
```

## Key Shapes

| Name | Shape | Meaning |
|---|---:|---|
| `B` | scalar | `config.batch_size`, default `256` |
| `N` | scalar | `config.num_sgd_steps_per_step`, default `64` |
| `B_total` | `B * N` | public batch from replay, default `16384` |
| `D` | scalar | `config.obs_dim`, state dimension |
| `G` | scalar | goal dimension |
| `A` | scalar | action dimension |
| `R` | scalar | `config.repr_dim`, default `64` |
| `transitions.observation` | `[B, D + G]` inner | `[state, goal]` |
| `transitions.action` | `[B, A]` | action |
| `critic logits` | `[B, B]` | all state-action vs all goals |

## Replay Batch Construction

File/function:

```text
contrastive.builder.ContrastiveBuilder.make_dataset_iterator.<locals>.flatten_fn
```

It takes one stored episode:

```text
sample.data.observation  [S, D + G]
sample.data.action       [S, A]
```

and makes relabeled transitions:

```text
state      = sample.data.observation[:-1, :D]
next_state = sample.data.observation[1:, :D]
goal       = future state converted by contrastive_utils.obs_to_goal_2d
new_obs    = concat([state, goal])
new_next_obs = concat([next_state, goal])
```

Mathematically:

```text
o_t = [s_t, h(s_k)]       where k > t
o_{t+1} = [s_{t+1}, h(s_k)]
```

## Critic Loss

File/function:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.critic_loss
```

Network call:

```text
logits, _, _ = networks.q_network.apply(q_params, observation, action)
```

Critic representation:

```text
sa_repr_i = phi(s_i, a_i)     [B, R]
g_repr_j  = psi(g_j)          [B, R]
logits[i, j] = dot(sa_repr_i, g_repr_j)
```

`I = jnp.eye(B)` marks positives:

```text
diagonal     i == j  positive
off-diagonal i != j  negative
```

Branches:

- `config.use_td=False`, `config.use_cpc=True`: CPC softmax loss.
- `config.use_td=False`, `config.use_cpc=False`: sigmoid BCE/NCE-style loss.
- `config.use_td=True`: TD/C-learning branch with target critic.

Gradient flow:

```text
critic_loss gradients -> q_params only
target_q_params are not directly optimized
```

## Actor Loss

File/function:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss
```

It samples actions:

```text
dist_params = policy_network.apply(policy_params, new_obs)
action = networks.sample(dist_params, key)
q_action, _, _ = q_network.apply(q_params, new_obs, action)
```

Objective:

```text
actor_loss = mean(-diag(q_action))
```

If entropy is enabled:

```text
actor_loss -= alpha * (-log_prob)
```

In normal `lp_contrastive.py::main`, `entropy_coefficient = 0.0`, so entropy has zero weight.

Gradient flow:

```text
actor_loss gradients -> policy_params
q_params provide the score but are not updated by actor_grad
```

## Parameter Updates

File/function:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.update_step
```

Critic:

```text
critic_grads = grad(critic_loss)(q_params, ...)
q_params = optax.apply_updates(q_params, critic_update)
```

Target critic:

```text
target_q_params = (1 - tau) * target_q_params + tau * q_params
```

Actor:

```text
actor_grads = grad(actor_loss)(policy_params, ...)
policy_params = optax.apply_updates(policy_params, actor_update)
```

## Logging

Important metrics:

- `critic_loss`
- `actor_loss`
- `binary_accuracy`
- `categorical_accuracy`
- `logits_pos`
- `logits_neg`
- `entropy_mean`
- `steps_per_second`

## Open Questions

1. `actor_loss` is called with `state.q_params`, not the just-updated local `q_params`.
2. The `twin_q=True` branch in `contrastive.networks.make_networks.<locals>._critic_fn` appears to reference `product`.
3. Exact metric aggregation behavior depends on `acme.jax.utils.process_multiple_batches`, external to this repo.

