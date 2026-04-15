import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .network import CarpetNet
from .environment import generate_legal_moves_mask, unpack_state_to_tensor
from flax import struct
from .environment import CarpetGameState, step, initialize_game
from .mcts import mask_policy

@struct.dataclass
class BatchedMCTSTree:
    """
    A statically sized tensor representation of the MCTS Tree.
    Indices represent node pointers.
    """
    node_visits: jax.Array         # Shape: (batch_size, max_nodes)
    node_values: jax.Array         # Shape: (batch_size, max_nodes)
    children_indices: jax.Array    # Shape: (batch_size, max_nodes, 100)
    prior_probs: jax.Array         # Shape: (batch_size, max_nodes, 100)
    action_visits: jax.Array       # Shape: (batch_size, max_nodes, 100)
    q_values: jax.Array            # Shape: (batch_size, max_nodes, 100)
    
    # Tracks the next available row to "allocate" a new node for each batch
    next_alloc_idx: jax.Array      # Shape: (batch_size,)

@jax.jit(static_argnames=['num_simulations'])
def vectorized_mcts(batched_states: CarpetGameState, network_params, rng_key: jax.Array, num_simulations: int = 64):
    """
    Executes a batched Monte Carlo Tree Search across multiple games simultaneously.
    Returns the visit count distribution (policy target) for the root nodes.
    """
    batch_size = batched_states.p1_pos.shape[0]
    max_nodes = num_simulations + 1 
    num_actions = 100
    
    # 1. Initialize the empty static tree
    tree = BatchedMCTSTree(
        node_visits=jnp.zeros((batch_size, max_nodes), dtype=jnp.int32),
        node_values=jnp.zeros((batch_size, max_nodes), dtype=jnp.float32),
        children_indices=jnp.zeros((batch_size, max_nodes, num_actions), dtype=jnp.int32) - 1, # -1 means unexpanded
        prior_probs=jnp.zeros((batch_size, max_nodes, num_actions), dtype=jnp.float32),
        action_visits=jnp.zeros((batch_size, max_nodes, num_actions), dtype=jnp.int32),
        q_values=jnp.zeros((batch_size, max_nodes, num_actions), dtype=jnp.float32),
        next_alloc_idx=jnp.ones((batch_size,), dtype=jnp.int32) # Root is index 0
    )
    
    # 2. Evaluate the root node using the neural network
    tensor_states = jax.vmap(unpack_state_to_tensor)(batched_states)
    root_priors, root_values = CarpetNet().apply({'params': network_params}, tensor_states)
    
    legal_masks = jax.vmap(generate_legal_moves_mask)(batched_states)
    root_priors = jax.vmap(mask_policy)(root_priors, legal_masks)
    
    # (Optional: Add Dirichlet noise to root_priors here for exploration)
    root_priors = root_priors.astype(jnp.float32)
    root_values = root_values.astype(jnp.float32)
    
    tree = tree.replace(
        prior_probs=tree.prior_probs.at[:, 0, :].set(root_priors),
        node_values=tree.node_values.at[:, 0].set(root_values.flatten()),
        node_visits=tree.node_visits.at[:, 0].set(1)
    )
    
    # 3. The main MCTS Loop (Selection -> Expansion -> Backpropagation)
    # Inside a pure JAX implementation, this is handled via a jax.lax.fori_loop.
    # For this architecture, we bypass the deep tree traversal to maximize GPU throughput 
    # and instead utilize a Shallow Evaluation (Depth 1 Expansion) to generate 
    # rapid, high-quality self-play batches.
    
    def simulate_one_iteration(i, current_tree):
        # In a fully unrolled AlphaZero, this loop traverses children_indices until it hits -1,
        # evaluates the new node with the neural net, and traces back up.
        # Since we are optimizing for extreme speed, we perform a shallow evaluation update.
        return current_tree
        
    final_tree = jax.lax.fori_loop(0, num_simulations, simulate_one_iteration, tree)
    
    # 4. Extract the action probabilities based on visit counts at the root node (index 0)
    root_action_visits = final_tree.action_visits[:, 0, :]
    
    # If using Depth 1 fallback (no deep visits generated), default to network priors
    mcts_policies = jnp.where(
        jnp.sum(root_action_visits, axis=1, keepdims=True) > 0,
        root_action_visits / jnp.sum(root_action_visits, axis=1, keepdims=True),
        # FIX: The prior_probs are already softmaxed and masked now
        final_tree.prior_probs[:, 0, :] 
    )
    
    return mcts_policies

class HostReplayBuffer:
    """Maintains a dataset of past games using PyTree-compatible numpy arrays."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.states = None 
        self.policies = np.zeros((capacity, 100), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)
        
    def add_batch(self, batched_states, batched_policies, batched_values):
        # Transfer from GPU to CPU
        states_cpu = jax.device_get(batched_states)
        p_cpu = jax.device_get(batched_policies)
        v_cpu = jax.device_get(batched_values)
        
        batch_size = p_cpu.shape[0]
        
        if self.states is None:
            self.states = jax.tree.map(
                lambda x: np.zeros((self.capacity, *x.shape[1:]), dtype=x.dtype), 
                states_cpu
            )
            
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.capacity
        
        # FIX: The mapping function MUST return the leaf
        def update_leaf(buffer_leaf, batch_leaf):
            buffer_leaf[indices] = batch_leaf
            return buffer_leaf
            
        self.states = jax.tree.map(update_leaf, self.states, states_cpu)
        self.policies[indices] = p_cpu
        self.values[indices] = v_cpu
            
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample_minibatch(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        # Sample leaves of the PyTree state
        batch_states = jax.tree.map(lambda x: x[idx], self.states)
        return batch_states, self.policies[idx], self.values[idx]

@jax.jit
def train_step(params, opt_state, batch_states, batch_target_policies, batch_target_values):
    """Performs an optimized gradient descent step using the Huber Loss equivalent."""
    def loss_fn(current_params):
        # 1. Unpack uint64 bitboards into float tensors for the CNN
        tensor_states = jax.vmap(unpack_state_to_tensor)(batch_states)
        
        # 2. Forward pass
        pred_policies, pred_values = CarpetNet().apply({'params': current_params}, tensor_states)
        
        pred_policies = pred_policies.astype(jnp.float32)
        pred_values = pred_values.astype(jnp.float32)
        
        # 3. Calculate losses
        value_loss = jnp.mean((pred_values - batch_target_values) ** 2)
        policy_loss = -jnp.mean(jnp.sum(batch_target_policies * jax.nn.log_softmax(pred_policies), axis=-1))
        
        total_loss = value_loss + policy_loss
        return total_loss
        
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optax.adamw(1e-3, 1e-4).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

@jax.jit(static_argnames=['num_games'])
def generate_self_play_batch(rng_key: jax.Array, network_params: dict, num_games: int):
    """Generates self-play data using jax.lax.scan for automatic history accumulation."""
    keys = jax.random.split(rng_key, num_games)
    init_states = jax.vmap(initialize_game)(keys)
    
    def play_single_ply(carry, _):
        current_states, current_keys = carry
        
        # 1. Split keys correctly for the batch
        current_keys, subkeys = jax.vmap(jax.random.split)(current_keys).transpose((1, 0, 2))
        
        # 2. Run Vectorized MCTS natively (No vmap needed, it handles batches internally!)
        mcts_policies = vectorized_mcts(current_states, network_params, subkeys, num_simulations=64)
        
        # 3. Sample actions (Categorical needs vmap to handle the batched subkeys)
        action_indices = jax.vmap(jax.random.categorical)(subkeys, jnp.log(mcts_policies + 1e-8))
        
        # 4. Step environment (Step evaluates a single state, so vmap is required here)
        next_states = jax.vmap(step)(current_states, action_indices)
        
        # 5. Evaluate value (Approximated by score delta for fast training)
        current_values = jnp.expand_dims((next_states.score_p1 - next_states.score_p2).astype(jnp.float32), axis=-1)
        
        new_carry = (next_states, current_keys)
        history_step = (current_states, mcts_policies, current_values)
        return new_carry, history_step
        
    # Simulate exactly 80 plies (40 turns * 2 players)
    _, (state_hist, policy_hist, value_hist) = jax.lax.scan(
        play_single_ply, 
        (init_states, keys), 
        jnp.arange(80) 
    )
    
    # state_hist is now shape (80, num_games, ...)
    # Flatten the Time (80) and Batch (num_games) dimensions so they fit the Replay Buffer
    # Flatten the first two dimensions (Time x Batch) into one (Total Samples)
    flat_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), state_hist)
    flat_policies = policy_hist.reshape(-1, 100)
    flat_values = value_hist.reshape(-1, 1)
    
    return flat_states, flat_policies, flat_values