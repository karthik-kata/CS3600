import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import struct

# Constants for edge masking in 64-bit space
UINT64_1 = jnp.uint64(1)
NOT_A_FILE = jnp.uint64(0xFEFEFEFEFEFEFEFE)
NOT_H_FILE = jnp.uint64(0x7F7F7F7F7F7F7F7F)

# Pre-calculated bit masks for fast tensor unpacking
SHIFT_ARRAY = jnp.arange(64, dtype=jnp.uint64)
BIT_MASK = jnp.left_shift(jnp.uint64(1), SHIFT_ARRAY)

@struct.dataclass
class CarpetGameState:
    """Represents the complete, fully-jittable game state."""
    blocked: jax.Array
    space: jax.Array
    primed: jax.Array
    carpeted: jax.Array
    p1_pos: jax.Array
    p2_pos: jax.Array
    score_p1: jax.Array
    score_p2: jax.Array
    turns_remaining: jax.Array
    rat_belief: jax.Array
    phantom_rat_pos: jax.Array

# --- Bitwise Shift Helpers ---
@jax.jit
def shift_north(b: jax.Array) -> jax.Array: 
    return jnp.right_shift(b, jnp.uint64(8))

@jax.jit
def shift_south(b: jax.Array) -> jax.Array: 
    return jnp.left_shift(b, jnp.uint64(8))

@jax.jit
def shift_east(b: jax.Array) -> jax.Array: 
    return jnp.left_shift(jnp.bitwise_and(b, NOT_H_FILE), jnp.uint64(1))

@jax.jit
def shift_west(b: jax.Array) -> jax.Array: 
    return jnp.right_shift(jnp.bitwise_and(b, NOT_A_FILE), jnp.uint64(1))

# --- State Tensorization ---
@jax.jit
def _unpack_single_bitboard(bitboard: jax.Array) -> jax.Array:
    """Converts a highly compressed 64-bit integer into an 8x8 array."""
    unpacked_1d = (jnp.bitwise_and(bitboard, BIT_MASK) > 0).astype(jnp.float32)
    return unpacked_1d.reshape((8, 8))

@jax.jit
def unpack_state_to_tensor(state: CarpetGameState) -> jax.Array:
    """Converts the internal bitboard state into a 7-channel neural network input."""
    c_blocked = _unpack_single_bitboard(state.blocked)
    c_space = _unpack_single_bitboard(state.space)
    c_primed = _unpack_single_bitboard(state.primed)
    c_carpeted = _unpack_single_bitboard(state.carpeted)
    
    # One-hot encode the worker positions
    c_p1 = (SHIFT_ARRAY == state.p1_pos).astype(jnp.float32).reshape((8, 8))
    c_p2 = (SHIFT_ARRAY == state.p2_pos).astype(jnp.float32).reshape((8, 8))
    
    # Reshape the rat belief
    c_rat = state.rat_belief.reshape((8, 8))
    
    # Stack along the channel axis
    return jnp.stack([c_blocked, c_space, c_primed, c_carpeted, c_p1, c_p2, c_rat], axis=-1)

# --- Legal Move Generation ---
@jax.jit
def generate_legal_moves_mask(state: CarpetGameState) -> jax.Array:
    """Generates a (100,) boolean array mapping exactly which moves are valid."""
    worker_mask = jnp.left_shift(jnp.uint64(1), state.p1_pos.astype(jnp.uint64))
    opponent_mask = jnp.left_shift(jnp.uint64(1), state.p2_pos.astype(jnp.uint64))
    impassable = jnp.bitwise_or(state.blocked, opponent_mask)
    
    # Check Plain Steps (Cannot step on impassable or primed)
    plain_invalid = jnp.bitwise_or(impassable, state.primed)
    plain_moves = jnp.array([
        (jnp.bitwise_and(shift_north(worker_mask), plain_invalid) == 0) & (shift_north(worker_mask) != 0),
        (jnp.bitwise_and(shift_east(worker_mask), plain_invalid) == 0) & (shift_east(worker_mask) != 0),
        (jnp.bitwise_and(shift_south(worker_mask), plain_invalid) == 0) & (shift_south(worker_mask) != 0),
        (jnp.bitwise_and(shift_west(worker_mask), plain_invalid) == 0) & (shift_west(worker_mask) != 0)
    ], dtype=jnp.bool_)
    
    # Check Prime Steps (Cannot prime impassable, already primed, or carpeted)
    prime_invalid = jnp.bitwise_or(jnp.bitwise_or(state.primed, state.carpeted), impassable)
    prime_moves = jnp.array([
        (jnp.bitwise_and(shift_north(worker_mask), prime_invalid) == 0) & (shift_north(worker_mask) != 0),
        (jnp.bitwise_and(shift_east(worker_mask), prime_invalid) == 0) & (shift_east(worker_mask) != 0),
        (jnp.bitwise_and(shift_south(worker_mask), prime_invalid) == 0) & (shift_south(worker_mask) != 0),
        (jnp.bitwise_and(shift_west(worker_mask), prime_invalid) == 0) & (shift_west(worker_mask) != 0)
    ], dtype=jnp.bool_)
    
    # ... (plain_moves and prime_moves definitions remain the same) ...
    
    def get_carpet_validity(shift_fn, start_mask, valid_carpet_board):
        m1 = shift_fn(start_mask)
        v1 = (jnp.bitwise_and(m1, valid_carpet_board) != 0)
        m2 = shift_fn(m1)
        v2 = v1 & (jnp.bitwise_and(m2, valid_carpet_board) != 0)
        m3 = shift_fn(m2)
        v3 = v2 & (jnp.bitwise_and(m3, valid_carpet_board) != 0)
        m4 = shift_fn(m3)
        v4 = v3 & (jnp.bitwise_and(m4, valid_carpet_board) != 0)
        m5 = shift_fn(m4)
        v5 = v4 & (jnp.bitwise_and(m5, valid_carpet_board) != 0)
        m6 = shift_fn(m5)
        v6 = v5 & (jnp.bitwise_and(m6, valid_carpet_board) != 0)
        m7 = shift_fn(m6)
        v7 = v6 & (jnp.bitwise_and(m7, valid_carpet_board) != 0)
        return jnp.array([v1, v2, v3, v4, v5, v6, v7], dtype=jnp.bool_)

    # FIX: Subtract both worker positions from the primed board to create a strictly "carpetable" board
    workers_mask = jnp.bitwise_or(worker_mask, opponent_mask)
    carpetable_board = jnp.bitwise_and(state.primed, jnp.bitwise_not(workers_mask))

    carpet_moves = jnp.concatenate([
        get_carpet_validity(shift_north, worker_mask, carpetable_board),
        get_carpet_validity(shift_east, worker_mask, carpetable_board),
        get_carpet_validity(shift_south, worker_mask, carpetable_board),
        get_carpet_validity(shift_west, worker_mask, carpetable_board)
    ])
    
    # Searches (Always valid)
    search_moves = jnp.ones(64, dtype=jnp.bool_)
    
    # Searches (Always valid)
    search_moves = jnp.ones(64, dtype=jnp.bool_)
    
    return jnp.concatenate([plain_moves, prime_moves, carpet_moves, search_moves])

# --- Hidden Markov Model Logic ---
NOISE_PROBS = jnp.array([
    [0.3,  0.5, 0.2],  # Blocked 
    [0.15, 0.7, 0.15], # Space 
    [0.8,  0.1, 0.1],  # Primed 
    [0.1,  0.1, 0.8]   # Carpet 
])

@jax.jit
def update_rat_belief(belief: jax.Array, transition_matrix: jax.Array, 
                      observed_noise: int, observed_distance: int,
                      worker_pos: int, floor_types: jax.Array) -> jax.Array:
    """Executes the exact Bayesian prediction and update steps for the rat's HMM."""
    predicted_belief = jnp.matmul(belief, transition_matrix)
    noise_likelihoods = NOISE_PROBS[floor_types, observed_noise]
    
    worker_y, worker_x = worker_pos // 8, worker_pos % 8
    all_y, all_x = jnp.arange(64) // 8, jnp.arange(64) % 8
    true_distances = jnp.abs(worker_y - all_y) + jnp.abs(worker_x - all_x)
    errors = observed_distance - true_distances
    
    distance_likelihoods = jnp.select(
        [errors == -1, errors == 0, errors == 1, errors == 2],
        [0.12, 0.7, 0.12, 0.06], default=0.0
    )
    
    updated_belief = predicted_belief * noise_likelihoods * distance_likelihoods
    return updated_belief / (jnp.sum(updated_belief) + 1e-8)

# --- Game Initialization ---
@jax.jit
def initialize_game(rng_key: jax.Array) -> CarpetGameState:
    """
    Generates a starting board state using purely compiled JAX operations.
    Places blocked squares in the corners and spawns workers mirrored in the center.
    """
    # 1. Randomly assign corner blocked configurations 
    # The rules specify 3x2, 2x3, or 2x2 in each corner.
    # For speed and to ensure XLA compilation, we can use a pre-calculated mask of 
    # a standard 2x3 corner layout for our self-play initialization.
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right base masks:
    blocked_mask = jnp.uint64(0x070700000000e0e0) 
    
    # 2. Spawn workers in the center 4x4 (Indices 26-29, 34-37)
    # Mirrored means if P1 is at (x, y), P2 is at (7-x, 7-y)
    p1_spawn_options = jnp.array([26, 27, 28, 29], dtype=jnp.int32)
    p1_pos = jax.random.choice(rng_key, p1_spawn_options)
    
    # Mirror calculation: flip x and y across the 8x8 grid
    p1_y, p1_x = p1_pos // 8, p1_pos % 8
    p2_pos = (7 - p1_y) * 8 + (7 - p1_x)
    
    # 3. Initialize HMM Uniform Belief
    initial_belief = jnp.ones(64, dtype=jnp.float32) / 64.0
    
    return CarpetGameState(
        blocked=blocked_mask,
        space=jnp.bitwise_not(blocked_mask), # All non-blocked are space
        primed=jnp.uint64(0),
        carpeted=jnp.uint64(0),
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        score_p1=jnp.int32(0),
        score_p2=jnp.int32(0),
        turns_remaining=jnp.int32(40),
        rat_belief=initial_belief,
        phantom_rat_pos=jnp.int32(0) # Will be sampled at MCTS root
    )

# --- Perspective Reversal ---
@jax.jit
def reverse_perspective(state: CarpetGameState) -> CarpetGameState:
    """
    Swaps the workers and scores so the neural network always evaluates 
    the board from the perspective of the current player.
    """
    return state.replace(
        p1_pos=state.p2_pos,
        p2_pos=state.p1_pos,
        score_p1=state.score_p2,
        score_p2=state.score_p1
    )

# --- The Core Transition Function ---
@jax.jit
def _apply_plain_step(state: CarpetGameState, direction_idx: jax.Array) -> CarpetGameState:
    p1_mask = jnp.left_shift(jnp.uint64(1), state.p1_pos.astype(jnp.uint64))
    
    # Ensure direction_idx is int32 for the switch
    shifted_mask = jax.lax.switch(
        direction_idx.astype(jnp.int32),
        [shift_north, shift_east, shift_south, shift_west],
        p1_mask
    )
    
    # argmax returns the 1D index of the set bit
    new_pos = jnp.argmax(jnp.bitwise_and(shifted_mask, BIT_MASK) > 0)
    
    return state.replace(
        p1_pos=new_pos.astype(jnp.int32), 
        turns_remaining=(state.turns_remaining - 1).astype(jnp.int32)
    )

@jax.jit
def _apply_prime_step(state: CarpetGameState, direction_idx: jax.Array) -> CarpetGameState:
    temp_state = _apply_plain_step(state, direction_idx)
    
    # The fix you identified, applied to the original position
    old_pos_mask = jnp.left_shift(jnp.uint64(1), state.p1_pos.astype(jnp.uint64))
    new_primed = jnp.bitwise_or(state.primed, old_pos_mask)
    
    return temp_state.replace(
        primed=new_primed, 
        score_p1=(state.score_p1 + 1).astype(jnp.int32)
    )

# Replace the old _apply_carpet_roll with this corrected version
@jax.jit
def _apply_carpet_roll(state: CarpetGameState, carpet_idx: jax.Array) -> CarpetGameState:
    dir_idx = carpet_idx // 7
    length = (carpet_idx % 7) + 1
    
    points_table = jnp.array([-1, 2, 4, 6, 10, 15, 21], dtype=jnp.int32)
    points_gained = points_table[length - 1]
    
    def apply_single_roll(i, carry):
        current_mask, accumulated_ray, player_pos_mask = carry
        next_mask = jax.lax.switch(
            dir_idx, 
            [shift_north, shift_east, shift_south, shift_west], 
            current_mask
        )
        valid_addition = jnp.where(i < length, next_mask, jnp.uint64(0))
        
        # FIX: Capture the player's position mask exactly where the carpet roll ends
        new_pos_mask = jnp.where(i == length - 1, next_mask, player_pos_mask)
        
        return next_mask, jnp.bitwise_or(accumulated_ray, valid_addition), new_pos_mask
        
    final_mask, ray_mask, final_pos_mask = jax.lax.fori_loop(
        0, 7, apply_single_roll, 
        (jnp.left_shift(UINT64_1, state.p1_pos.astype(jnp.uint64)), jnp.uint64(0), jnp.uint64(0))
    )
    
    new_primed = jnp.bitwise_and(state.primed, jnp.bitwise_not(ray_mask))
    new_carpeted = jnp.bitwise_or(state.carpeted, ray_mask)
    
    # FIX: Use the captured final_pos_mask, not the 7-shift final_mask
    new_pos = jnp.argmax(jnp.bitwise_and(final_pos_mask, BIT_MASK) > 0)
    
    return state.replace(
        p1_pos=new_pos.astype(jnp.int32), 
        primed=new_primed, 
        carpeted=new_carpeted, 
        score_p1=state.score_p1 + points_gained
    )

@jax.jit
def step(state: CarpetGameState, action_idx: jax.Array) -> CarpetGameState:
    def do_plain(idx): return _apply_plain_step(state, idx)
    def do_prime(idx): return _apply_prime_step(state, idx - 4)
    def do_carpet(idx): return _apply_carpet_roll(state, idx - 8)
    
    def do_search(idx): 
        search_pos = idx - 36
        # Ensure the comparison is against an integer
        points = jnp.where(search_pos == state.phantom_rat_pos.astype(jnp.int32), 4, -2) 
        return state.replace(score_p1=(state.score_p1 + points).astype(jnp.int32))
    
    category = jnp.select(
        [action_idx < 4, action_idx < 8, action_idx < 36], 
        [0, 1, 2], 
        default=3
    )
    
    next_state = jax.lax.switch(category, [do_plain, do_prime, do_carpet, do_search], action_idx)
    
    # FINAL CLEANUP: Ensure every integer field stays an integer for the next iteration
    return reverse_perspective(next_state.replace(
        p1_pos=next_state.p1_pos.astype(jnp.int32),
        p2_pos=next_state.p2_pos.astype(jnp.int32),
        score_p1=next_state.score_p1.astype(jnp.int32),
        score_p2=next_state.score_p2.astype(jnp.int32),
        turns_remaining=(next_state.turns_remaining - 1).astype(jnp.int32)
    ))

# (Retain unpack_state_to_tensor, generate_legal_moves_mask, and update_rat_belief here)