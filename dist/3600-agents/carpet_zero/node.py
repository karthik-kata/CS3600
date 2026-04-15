import math
from typing import Dict, List, Optional, Tuple
from game.move import Move

class MCTSNode:
    """
    A memory-optimized Node for the AlphaZero Monte Carlo Tree Search.
    Uses __slots__ to drastically reduce memory footprint during deep rollouts.
    """
    __slots__ = (
        'parent', 'move', 'children', 
        'visit_count', 'value_sum', 'prior_prob'
    )

    def __init__(self, parent: Optional['MCTSNode'], prior_prob: float, move: Optional[Move] = None):
        """
        Args:
            parent: The parent MCTSNode (None if this is the root).
            prior_prob: The probability of reaching this node, given by the NN policy.
            move: The move that led to this node (None if this is the root).
        """
        self.parent = parent
        self.move = move
        self.children: Dict[int, 'MCTSNode'] = {} # Keyed by action_index
        
        # MCTS Statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior_prob: float = prior_prob

    @property
    def q_value(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Returns True if the node has been expanded (has children)."""
        return len(self.children) > 0

    def expand(self, action_probs: List[Tuple[int, float, Move]]):
        """
        Expands the node by creating children for all valid moves.
        
        Args:
            action_probs: A list of tuples containing (action_index, probability, Move object).
                          These probabilities come directly from the Neural Network.
        """
        for action_idx, prob, move in action_probs:
            if action_idx not in self.children:
                self.children[action_idx] = MCTSNode(
                    parent=self, 
                    prior_prob=prob, 
                    move=move
                )

    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'MCTSNode']:
        """
        Selects the child with the highest PUCT score.
        
        Args:
            c_puct: Exploration constant. Higher values encourage exploring unvisited nodes.
            
        Returns:
            A tuple of (action_index, selected_child_node).
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # Optimization: Precompute the square root of the parent's visit count
        # This prevents calculating math.sqrt() repeatedly inside the loop
        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action_idx, child in self.children.items():
            # Calculate the Upper Confidence Bound
            u_value = c_puct * child.prior_prob * (sqrt_parent_visits / (1 + child.visit_count))
            
            # PUCT score = Q + U
            puct_score = child.q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_action = action_idx
                best_child = child

        return best_action, best_child

    def backpropagate(self, value: float):
        """
        Updates the node's statistics recursively up to the root.
        
        Args:
            value: The evaluation score from the Neural Network (from the perspective of the player who just moved).
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Note: In zero-sum alternating games, the value flips for the parent.
            # E.g., a +1 win for Player A is a -1 loss for Player B.
            self.parent.backpropagate(-value)