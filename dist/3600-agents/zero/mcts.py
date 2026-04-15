import jax
import jax.numpy as jnp
import time

def calculate_turn_budget(turn_number: int, total_time_remaining: float) -> float:
    """Dynamically front-loads compute time to complex early game states."""
    turns_left = 40 - turn_number
    if total_time_remaining < 20.0:
        return total_time_remaining / max(1, turns_left)
    
    if turn_number <= 15: weight = 3.0
    elif turn_number <= 30: weight = 2.0
    else: weight = 1.0
        
    remaining_weights = sum(3.0 if t <= 15 else (2.0 if t <= 30 else 1.0) for t in range(turn_number, 41))
    return ((weight / remaining_weights) * total_time_remaining) - 0.1

@jax.jit
def mask_policy(raw_policy_logits: jax.Array, legal_mask: jax.Array) -> jax.Array:
    """Filters out illegal moves, forcing the network to only consider valid strategies."""
    masked_logits = jnp.where(legal_mask, raw_policy_logits, -jnp.inf)
    return jax.nn.softmax(masked_logits)

@jax.jit
def puct_selection(q_values: jax.Array, visit_counts: jax.Array, prior_probs: jax.Array, valid_moves_mask: jax.Array, c_puct: float = 1.5) -> jax.Array:
    """Evaluates the UCB1 metric (PUCT) to balance exploration vs. exploitation."""
    total_visits = jnp.sum(visit_counts)
    exploration_term = c_puct * prior_probs * (jnp.sqrt(total_visits) / (1.0 + visit_counts))
    puct_scores = q_values + exploration_term
    masked_scores = jnp.where(valid_moves_mask, puct_scores, -jnp.inf)
    return jnp.argmax(masked_scores)

@jax.jit
def sample_phantom_rat(rng_key: jax.Array, rat_belief: jax.Array) -> jax.Array:
    """Samples a concrete rat position from the HMM to enable fast JAX determinism."""
    return jax.random.choice(rng_key, a=jnp.arange(64, dtype=jnp.int32), p=rat_belief)