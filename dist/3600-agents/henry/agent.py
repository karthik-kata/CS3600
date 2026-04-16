import numpy as np
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums
from .hmm import RatHMM
from .expectiminimax import get_best_move

class PlayerAgent:
    """
    The main tournament agent that integrates the Hidden Markov Model 
    and Expectiminimax search to play the game optimally.
    """

    def __init__(self, board: board.Board, transition_matrix: np.ndarray = None, time_left: Callable = None):
        """
        Initializes the agent and sets up the HMM for tracking the rat.
        """
        self.hmm = RatHMM(transition_matrix)
        self.is_first_turn = True

    def commentate(self) -> str:
        """
        Optional: Prints out commentary at the end of the game.
        """
        return "Good game :D"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> move.Move:
        """
        Executes the bot's turn logic:
        1. Synchronizes the HMM with the unobserved rat steps.
        2. Applies the current turn's sensor data to the HMM.
        3. Budgets time and calls the Expectiminimax search.
        """
        noise, estimated_distance = sensor_data
        my_worker = board.player_worker
        my_loc = my_worker.get_location()

        # --- 1. Synchronize the Hidden Markov Model ---
        unobserved_steps = 0
        
        # Check if the rat was caught and respawned
        if board.opponent_search[1]:
            # Opponent caught it on their turn. It respawned, took 1000 steps. 
            # 0 unobserved regular steps between respawn and now.
            self.hmm.reset()
        elif board.player_search[1]:
            # We caught it on our last turn. It respawned, took 1000 steps.
            # It moved 1 unobserved step before the opponent's turn.
            self.hmm.reset()
            unobserved_steps = 1
        else:
            # Nobody caught it recently.
            if self.is_first_turn:
                # If we are Player B (turn_count == 1), the rat moved once for Player A's turn.
                if board.turn_count == 1:
                    unobserved_steps = 1
                else:
                    unobserved_steps = 0
            else:
                # Normal subsequent turn: the rat moved once before the opponent's turn.
                unobserved_steps = 1

        # Apply the unobserved steps without sensor data
        for _ in range(unobserved_steps):
            self.hmm.belief = np.dot(self.hmm.belief, self.hmm.T)

        self.is_first_turn = False

        # --- 2. Predict current turn and Update with Sensor Data ---
        self.hmm.predict_and_update(board, my_loc, noise, estimated_distance)

        # --- 3. Time Budgeting ---
        # Allocate roughly proportional time based on remaining turns, leaving a buffer
        remaining_time = time_left()
        turns_left = max(my_worker.turns_left, 1)
        
        # Aim to use an equal fraction of the remaining time, but cap it so we 
        # don't waste 20 seconds on a single move if we have a surplus.
        allocated_time = (remaining_time / turns_left) * 0.9
        allocated_time = min(allocated_time, 8.0) 
        
        # Hard safety margin to prevent losing via timeout
        allocated_time = min(allocated_time, remaining_time - 0.5)

        # --- 4. Search and Execute ---
        best_move = get_best_move(
            board=board,
            max_time=allocated_time,
            is_player_a=my_worker.is_player_a,
            rat_belief=self.hmm.belief
        )

        return best_move