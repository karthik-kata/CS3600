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

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable) -> move.Move:
            noise, estimated_distance = sensor_data
            my_worker = board.player_worker
            my_loc = my_worker.get_location()

            # --- 1. Synchronize the Hidden Markov Model ---
            player_searched = board.player_search[0] is not None
            player_caught = board.player_search[1] if player_searched else False
            
            opp_searched = board.opponent_search[0] is not None
            opp_caught = board.opponent_search[1] if opp_searched else False
            
            if opp_caught:
                self.hmm.reset()
            elif player_caught:
                self.hmm.reset()
                # FIX: Advance rat 1 step FOR the opponent's turn, THEN register their miss
                self.hmm.belief = np.dot(self.hmm.belief, self.hmm.T)
                if opp_searched and not opp_caught:
                    self.hmm.register_miss(board.opponent_search[0])
            else:
                if self.is_first_turn:
                    if board.turn_count == 1: 
                        # FIX: Advance rat, then register miss
                        self.hmm.belief = np.dot(self.hmm.belief, self.hmm.T)
                        if opp_searched and not opp_caught:
                            self.hmm.register_miss(board.opponent_search[0])
                else:
                    # FIX: Advance rat, then register miss
                    self.hmm.belief = np.dot(self.hmm.belief, self.hmm.T)
                    if opp_searched and not opp_caught:
                        self.hmm.register_miss(board.opponent_search[0])

            self.is_first_turn = False

            # --- 2. Predict current turn and Update with Sensor Data ---
            self.hmm.predict_and_update(board, my_loc, noise, estimated_distance)

            # --- 3. Time Budgeting ---
            remaining_time = time_left()
            turns_left = max(my_worker.turns_left, 1)
            allocated_time = min((remaining_time / turns_left) * 0.9, 8.0) 
            allocated_time = min(allocated_time, remaining_time - 0.5)

            # --- 4. Search and Execute ---
            best_move = get_best_move(
                board=board,
                max_time=allocated_time,
                is_player_a=my_worker.is_player_a,
                rat_belief=self.hmm.belief,
                respawn_belief=self.hmm.default_respawn_belief,
                hmm_trans = self.hmm.T
            )

            return best_move