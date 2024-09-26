from typing import List, Optional
from .heuristic import calculate_manhattan, calculate_exponential
goal_state = [
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2]
]

class State:
    def __init__(self, state: List[List[int]], parent: Optional['State'], action: Optional[List[List[int]]], heuristic: int, cost: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.cost = cost

    def __lt__(self, other: 'State') -> bool:
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def is_goal(state: List[List[int]]) -> bool:
    return state == goal_state

def get_successors(state: List[List[int]], heuristic_type: str) -> List[State]:
    successors = []
    dx = [0, 0, 1, -1]  # Horizontal moves: right, left
    dy = [-1, 1, 0, 0]  # Vertical moves: down, up

    for i in range(7):
        for j in range(7):
            if state[i][j] == 1:  # Current position of a peg
                for k in range(4):  # Check all four possible directions
                    new_i = i + dy[k] * 2  # Move two in the current direction
                    new_j = j + dx[k] * 2  # Move two in the current direction
                    mid_i = i + dy[k]  # Position of the peg to jump over
                    mid_j = j + dx[k]  # Position of the peg to jump over

                    # Check if the move is valid
                    if (0 <= new_i < 7 and 0 <= new_j < 7 and
                        state[mid_i][mid_j] == 1 and state[new_i][new_j] == 0):
                        # Create new state by making the move
                        new_state = [row[:] for row in state]
                        new_state[i][j] = 0  # Current peg moves to empty
                        new_state[mid_i][mid_j] = 0  # Jumped over peg is removed
                        new_state[new_i][new_j] = 1  # New position for the peg

                        heuristic = calculate_manhattan(new_state) if heuristic_type == "Manhattan" else calculate_exponential(new_state)
                        successors.append(State(new_state, None, [[i, j], [new_i, new_j]], heuristic, 1))

    return successors

def calculate_heuristic(state: List[List[int]], heuristic_type: str) -> int:
    return calculate_manhattan(state) if heuristic_type == "Manhattan" else calculate_exponential(state)

def construct_path(state: State) -> List[List[List[int]]]:
    path = []
    while state.parent:
        path.append(state.action)
        state = state.parent
    return list(reversed(path))
