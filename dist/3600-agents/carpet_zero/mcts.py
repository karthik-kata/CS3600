import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from game.board import Board
from game.enums import MoveType
from game.move import Move

# --- POLYGLOT IMPORT FIX ---
try:
    from .node import MCTSNode
    from .serializer import StateSerializer
    from .model import index_to_move
except (ImportError, ValueError):
    from node import MCTSNode
    from serializer import StateSerializer
    from model import index_to_move

class AlphaZeroMCTS:
    """
    The core Monte Carlo Tree Search algorithm guided by a Neural Network.
    Optimized for batched SLURM cluster self-play execution.
    """
    def __init__(self, model: torch.nn.Module, serializer: 'StateSerializer', 
                 num_simulations: int = 400, c_puct: float = 1.0, 
                 temperature: float = 1.0):
        """
        Args:
            model: The trained CarpetZeroNet PyTorch model.
            serializer: The StateSerializer instance.
            num_simulations: How many MCTS iterations to run per move.
            c_puct: The PUCT exploration constant.
            temperature: Controls exploration in the final move selection.
        """
        self.model = model
        self.serializer = serializer
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _evaluate_and_expand(self, node: 'MCTSNode', board: Board, hmm_belief: np.ndarray):
        """
        Uses the Neural Network to evaluate the leaf node and expand it with valid moves.
        """
        # 1. Serialize the state (no HMM belief required)
        spatial_tensor, scalar_tensor = self.serializer.serialize_single(board, hmm_belief)
        spatial_tensor = spatial_tensor.to(self.device)
        scalar_tensor = scalar_tensor.to(self.device)

        # 2. Neural Network Inference
        policy_logits, value = self.model(spatial_tensor, scalar_tensor)
        
        # Squeeze batch dimension
        policy_logits = policy_logits.squeeze(0)
        value = value.item()

        # 3. Mask Invalid Moves (strictly excludes search actions)
        valid_moves_enums = board.get_valid_moves(exclude_search=True)
        
        # We need to map the valid Move objects back to their network indices [0, 35]
        valid_indices = []
        valid_moves = []
        for move in valid_moves_enums:
            idx = self._move_to_index(move)
            valid_indices.append(idx)
            valid_moves.append(move)

        if not valid_indices:
            # Terminal state or no valid moves
            return value

        # Create a mask (-inf for invalid moves) before softmax
        mask = torch.ones_like(policy_logits, dtype=torch.bool)
        mask[valid_indices] = False
        policy_logits[mask] = -float('inf')

        # 4. Apply Softmax to get probabilities
        action_probs = F.softmax(policy_logits, dim=0)

        # 5. Expand the node
        expansion_data = []
        for idx, move in zip(valid_indices, valid_moves):
            prob = action_probs[idx].item()
            expansion_data.append((idx, prob, move))
            
        node.expand(expansion_data)
        
        return value

    def search(self, initial_board: Board, initial_hmm_belief: np.ndarray) -> Tuple[Move, np.ndarray]:
        """
        Executes the MCTS loop and returns the best move and the policy distribution.
        """
        root = MCTSNode(parent=None, prior_prob=1.0)
        
        # Evaluate and expand the root node immediately
        self._evaluate_and_expand(root, initial_board, initial_hmm_belief)

        # Add Dirichlet noise to the root node to encourage exploration during self-play
        valid_actions = list(root.children.keys())
        if valid_actions:  # Safety check in case of no valid moves
            noise = np.random.dirichlet([0.3] * len(valid_actions))
            for i, action_idx in enumerate(valid_actions):
                # 75% NN Prior, 25% Dirichlet Noise
                root.children[action_idx].prior_prob = 0.75 * root.children[action_idx].prior_prob + 0.25 * noise[i]

        # Core MCTS Loop
        for _ in range(self.num_simulations):
            node = root
            # We create a lightweight copy of the board to simulate moves forward
            sim_board = initial_board.get_copy(build_history=False)
            
            # Phase 1: Selection (Traverse down the tree using PUCT)
            while node.is_expanded():
                action_idx, node = node.select_child(self.c_puct)
                # Apply the move to our simulated board state
                sim_board.apply_move(node.move, timer=0, check_ok=False)
                # Flip perspective for the next player
                sim_board.reverse_perspective()
            
            # Phase 2 & 3: Evaluation and Expansion
            value = self._evaluate_and_expand(node, sim_board, initial_hmm_belief)

            # Phase 4: Backpropagation
            # If the perspective was flipped an odd number of times, we must negate the value
            node.backpropagate(-value)

        # Action Selection based on visit counts (36 dimensions)
        action_visits = np.zeros(36, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_visits[action_idx] = child.visit_count

        if self.temperature == 0:
            # Deterministic selection (Tournament mode)
            best_action = np.argmax(action_visits)
            policy_target = np.zeros_like(action_visits)
            policy_target[best_action] = 1.0
        else:
            # Stochastic selection (Self-play training mode)
            
            # --- NEW NUMERICAL STABILITY FIX ---
            max_visits = np.max(action_visits)
            if max_visits > 0:
                # Scale everything down to [0.0, 1.0] before applying the massive exponent
                action_visits = action_visits / max_visits             
            action_visits = action_visits ** (1.0 / self.temperature)
            policy_sum = np.sum(action_visits)
            if policy_sum > 0:
                policy_target = action_visits / policy_sum
            else:
                # Fallback to uniform distribution if something goes wrong
                policy_target = np.ones(36, dtype=np.float32) / 36.0
            best_action = np.random.choice(36, p=policy_target)

        return root.children[best_action].move, policy_target

    def _move_to_index(self, move: Move) -> int:
        """Helper to map a Move object back to its 0-35 network index."""
        if move.move_type == MoveType.PLAIN:
            return int(move.direction)
        elif move.move_type == MoveType.PRIME:
            return 4 + int(move.direction)
        elif move.move_type == MoveType.CARPET:
            return 8 + (int(move.direction) * 7) + (move.roll_length - 1)
        return 0