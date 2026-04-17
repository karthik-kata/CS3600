import numpy as np
from typing import Tuple
from game.enums import Cell, Noise, BOARD_SIZE

class RatHMM:
    """
    Hidden Markov Model to track the unobservable rat position on the 8x8 grid.
    Maintains a probability distribution (belief) of the rat's location.
    """
    def __init__(self, transition_matrix: np.ndarray):
        self.T = transition_matrix
        self.num_states = BOARD_SIZE * BOARD_SIZE
        self.belief = np.zeros(self.num_states, dtype=np.float64)
        
        # Pre-compute the 1000-step belief vector starting from (0,0).
        # Iterating the belief vector is much faster than matrix_power on a
        # full 64x64 matrix (O(n) dot products vs O(n^3 log k)).
        _init = np.zeros(self.num_states, dtype=np.float64)
        _init[0] = 1.0
        for _ in range(1000):
            _init = np.dot(_init, self.T)
        self.belief_after_spawn = _init
        
        # Noise emission probabilities: P(noise | cell_type)
        # Ordered as (Squeak, Scratch, Squeal) corresponding to enums.Noise
        self.noise_probs = {
            Cell.BLOCKED: (0.5, 0.3, 0.2),
            Cell.SPACE: (0.7, 0.15, 0.15),
            Cell.PRIMED: (0.1, 0.8, 0.1),
            Cell.CARPET: (0.1, 0.1, 0.8),
        }
        
        # Distance error probabilities: P(offset) where offset = rolled_dist - actual_dist
        self.dist_error_probs = {
            -1: 0.12,
            0: 0.7,
            1: 0.12,
            2: 0.06
        }
        self.dist_offsets = [-1, 0, 1, 2]

        self.reset()

    def reset(self):
        """
        Resets the belief state when a new rat is spawned (at game start or after capture).
        The rat spawns at (0, 0) and instantly takes 1000 hidden steps.
        """
        self.belief = self.belief_after_spawn.copy()

    def predict_and_update(self, board, player_pos: Tuple[int, int], noise: Noise, distance_estimate: int):
        """
        Predicts the rat's movement for the current turn and updates belief based on sensor data.
        """
        # Step 1: Predict (Rat moves 1 square before each turn)
        self.belief = np.dot(self.belief, self.T)

        # Step 2: Update (Apply observation likelihoods)
        likelihoods = np.zeros(self.num_states, dtype=np.float64)

        for i in range(self.num_states):
            x = i % BOARD_SIZE
            y = i // BOARD_SIZE
            loc = (x, y)

            # --- 1. Noise Likelihood ---
            cell_type = board.get_cell(loc)
            # Default to SPACE if somehow an invalid type is queried
            cell_noise_distribution = self.noise_probs.get(cell_type)
            noise_prob = cell_noise_distribution[noise]

            # --- 2. Distance Likelihood ---
            actual_dist = abs(player_pos[0] - x) + abs(player_pos[1] - y)
            dist_prob = 0.0
            
            if distance_estimate == 0:
                # If the observed estimate is 0, the internal rolled distance was <= 0
                for offset in self.dist_offsets:
                    if actual_dist + offset <= 0:
                        dist_prob += self.dist_error_probs[offset]
            else:
                # If the observed estimate > 0, it exactly equals the actual + offset
                offset = distance_estimate - actual_dist
                dist_prob = self.dist_error_probs.get(offset, 0.0)

            # Combined Likelihood for this cell
            likelihoods[i] = noise_prob * dist_prob

        # Apply likelihoods to belief state
        self.belief *= likelihoods

        # Step 3: Normalize the distribution
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            # Fallback uniform distribution in case of numerical underflow anomalies
            self.belief = np.ones(self.num_states, dtype=np.float64) / self.num_states

    def get_most_likely_position(self) -> Tuple[Tuple[int, int], float]:
        """
        Returns the (x, y) coordinate with the highest probability and its probability value.
        """
        best_idx = int(np.argmax(self.belief))
        prob = self.belief[best_idx]
        x = best_idx % BOARD_SIZE
        y = best_idx // BOARD_SIZE
        return (x, y), float(prob)

    def get_prob(self, loc: Tuple[int, int]) -> float:
        """
        Returns the specific probability of the rat being at a given location.
        """
        idx = loc[1] * BOARD_SIZE + loc[0]
        return float(self.belief[idx])