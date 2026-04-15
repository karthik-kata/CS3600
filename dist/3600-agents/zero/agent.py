import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pickle
import os
import time

# Correct engine imports
from game.enums import MoveType, Direction
from game.move import Move

# Local relative imports
from .network import CarpetNet
from .environment import CarpetGameState, update_rat_belief, _unpack_single_bitboard, generate_legal_moves_mask
from .mcts import calculate_turn_budget
from .pipeline import vectorized_mcts

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left=None):
        weights_path = os.path.join(os.path.dirname(__file__), "weights.pkl")
        rng = jax.random.PRNGKey(0)
        
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                data = pickle.load(f)
                self.params = data['params'] if isinstance(data, dict) and 'params' in data else data
        else:
            # FIX: Initialize default weights if no checkpoint is found
            dummy_input = jnp.zeros((1, 8, 8, 7))
            self.params = CarpetNet().init(rng, dummy_input)['params']
            
        # Double-check that we have the weights, not a variables dict
        if hasattr(self, 'params') and isinstance(self.params, dict) and 'params' in self.params:
             self.params = self.params['params']
                 
        # Double-check that we have the weights, not a variables dict
        if isinstance(self.params, dict) and 'params' in self.params:
             self.params = self.params['params']
            
        self.net = CarpetNet()
        self.rat_belief = jnp.ones(64, dtype=jnp.float32) / 64.0
        
        # Use the transition matrix provided by the engine
        self.transition_matrix = transition_matrix if transition_matrix is not None else jnp.ones((64, 64)) / 64.0 
        
        # Warmup XLA (Ensure num_simulations matches your play loop for faster first turns)
        dummy_state = self._translate_board_to_jax(board)
        vectorized_mcts(dummy_state, self.params, rng, num_simulations=16)

    # FIX: Rename get_move to play to match the tournament runner
    def play(self, board, rat_signal, get_time_remaining):
        noise, distance = rat_signal
        turn_number = board.turn_count #
        time_budget = calculate_turn_budget(turn_number, get_time_remaining())
        
        jax_state = self._translate_board_to_jax(board)
        
        # FIX: Dynamically derive floor types for the Bayesian update
        # NOISE_PROBS indices: 0=Blocked, 1=Space, 2=Primed, 3=Carpet
        blocked_tiles = _unpack_single_bitboard(jax_state.blocked[0]).flatten()
        primed_tiles = _unpack_single_bitboard(jax_state.primed[0]).flatten()
        carpeted_tiles = _unpack_single_bitboard(jax_state.carpeted[0]).flatten()
        
        floor_types = jnp.ones(64, dtype=jnp.int32) # Default board to Space (1)
        floor_types = jnp.where(primed_tiles, 2, floor_types)
        floor_types = jnp.where(carpeted_tiles, 3, floor_types)
        floor_types = jnp.where(blocked_tiles, 0, floor_types) # Blocked overwrites
        
        self.rat_belief = update_rat_belief(
            self.rat_belief, self.transition_matrix, 
            noise, distance, jax_state.p1_pos[0], floor_types
        )
        
        jax_state = jax_state.replace(rat_belief=jnp.expand_dims(self.rat_belief, 0))
        
        # Execute MCTS chunks within the time budget
        start_time = time.time()
        aggregated_policies = jnp.zeros(100, dtype=jnp.float32)
        rng_key = jax.random.PRNGKey(int(time.time() * 1000))
        
        while time.time() - start_time < time_budget:
            rng_key, subkey = jax.random.split(rng_key)
            chunk_policy = vectorized_mcts(jax_state, self.params, subkey, num_simulations=16)
            aggregated_policies += chunk_policy[0]
        
        legal_mask = generate_legal_moves_mask(jax_state)
            
        masked_policies = jnp.where(legal_mask, aggregated_policies, -jnp.inf)
        best_action_idx = int(jnp.argmax(masked_policies))
        
        return self._idx_to_move(best_action_idx)

    # (Retain your _translate_board_to_jax and _idx_to_move methods below)

    def _translate_board_to_jax(self, board) -> CarpetGameState:
        """Extracts the internal masks and variables from the tournament Board object."""
        
        # Extract the 64-bit integer masks and convert them to JAX uint64 types
        blocked_mask = jnp.uint64(board._blocked_mask)
        space_mask = jnp.uint64(board._space_mask)
        primed_mask = jnp.uint64(board._primed_mask)
        carpet_mask = jnp.uint64(board._carpet_mask)

        # Convert 2D tuple locations into 1D flat indices (0-63)
        p1_x, p1_y = board.player_worker.get_location()
        p1_idx = jnp.int32(p1_y * 8 + p1_x)

        p2_x, p2_y = board.opponent_worker.get_location()
        p2_idx = jnp.int32(p2_y * 8 + p2_x)

        # Extract scores and turns remaining
        score_p1 = jnp.int32(board.player_worker.get_points())
        score_p2 = jnp.int32(board.opponent_worker.get_points())
        turns_rem = jnp.int32(board.player_worker.turns_left)

        return CarpetGameState(
            blocked=jnp.array([blocked_mask], dtype=jnp.uint64),
            space=jnp.array([space_mask], dtype=jnp.uint64),
            primed=jnp.array([primed_mask], dtype=jnp.uint64),
            carpeted=jnp.array([carpet_mask], dtype=jnp.uint64),
            p1_pos=jnp.array([p1_idx], dtype=jnp.int32),
            p2_pos=jnp.array([p2_idx], dtype=jnp.int32),
            score_p1=jnp.array([score_p1], dtype=jnp.int32),
            score_p2=jnp.array([score_p2], dtype=jnp.int32),
            turns_remaining=jnp.array([turns_rem], dtype=jnp.int32),
            rat_belief=jnp.expand_dims(self.rat_belief, axis=0),
            phantom_rat_pos=jnp.array([0], dtype=jnp.int32) 
        )

    def _idx_to_move(self, idx: int) -> Move:
        """Translates the network's 100-dim action array back to a game Move object."""
        
        # Helper list to map indices to the Direction enum
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        
        # 1. Plain Steps (Indices 0 - 3)
        if idx < 4:
            return Move.plain(directions[idx])
            
        # 2. Prime Steps (Indices 4 - 7)
        elif idx < 8:
            return Move.prime(directions[idx - 4])
            
        # 3. Carpet Rolls (Indices 8 - 35)
        elif idx < 36:
            carpet_idx = idx - 8
            dir_idx = carpet_idx // 7  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            length = (carpet_idx % 7) + 1 # Lengths 1 through 7
            return Move.carpet(directions[dir_idx], length)
            
        # 4. Search Moves (Indices 36 - 99)
        else:
            search_idx = idx - 36
            search_y = search_idx // 8
            search_x = search_idx % 8
            return Move.search((search_x, search_y))