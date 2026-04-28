# Algorithm Variants Short

## Key Idea

The `--alg` flag mostly toggles config booleans:

```text
use_cpc
use_td
twin_q
add_mc_to_td
```

The loss branches live in:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.critic_loss
```

The flag dispatch lives in:

```text
lp_contrastive.py::main
```

## Comparison Table

| `--alg` | Accepted? | Config set in `lp_contrastive.py::main` | Critic loss |
|---|---|---|---|
| `contrastive_nce` | No | None; raises `NotImplementedError` | Latent branch would be sigmoid BCE with `use_td=False`, `use_cpc=False` |
| `contrastive_cpc` | Yes | `use_cpc=True` | softmax cross-entropy over `[B, B]` logits |
| `c_learning` | Yes | `use_td=True`, `twin_q=True` | TD/C-learning BCE with target critic |
| `nce+c_learning` | Yes | `use_td=True`, `twin_q=True`, `add_mc_to_td=True` | TD/C-learning with mixed goals and changed coefficients |

## Shared Actor Objective

File/function:

```text
contrastive.learning.ContrastiveLearner.__init__.<locals>.actor_loss
```

All variants use:

```text
action ~ pi(. | state, goal)
q_action = Q(state, action, goal)
actor_loss = mean(-diag(q_action))
```

If `config.use_action_entropy=True`, code adds:

```text
- alpha * (-log_prob)
```

But `lp_contrastive.py::main` sets:

```text
entropy_coefficient = 0.0
```

so entropy has zero weight in normal entry-point runs.

## `contrastive_cpc`

Config:

```text
use_cpc=True
use_td=False
twin_q=False
```

Critic objective:

```text
softmax_cross_entropy(logits, labels=I)
+ 0.01 * logsumexp(logits, axis=1)^2
```

Mental model:

```text
For each state-action row, pick the correct goal from all goals in the batch.
```

## `c_learning`

Config:

```text
use_td=True
twin_q=True
add_mc_to_td=False
```

Critic objective uses:

```text
target_q_params
next_action ~ policy(next_observation)
next_v = stop_gradient(diag(min(sigmoid(next_q), axis=-1)))
w = clip(next_v / (1 - next_v), 0, 20)
```

Loss:

```text
(1 - gamma) * loss_pos
+ gamma * loss_neg1
+ loss_neg2
```

## `nce+c_learning`

Config:

```text
use_td=True
twin_q=True
add_mc_to_td=True
```

Difference from `c_learning`:

```text
mixes obs_to_goal(next_s) with existing relabeled goals
uses different TD loss coefficients
```

Loss:

```text
(1 + (1 - gamma)) * loss_pos
+ gamma * loss_neg1
+ 2 * loss_neg2
```

## Open Questions

1. `contrastive_nce` is documented but not accepted by `lp_contrastive.py::main`.
2. `nce+c_learning` does not visibly add the non-TD NCE loss; it modifies the TD branch.
3. `twin_q=True` depends on the `_critic_fn` branch that references `product`.

