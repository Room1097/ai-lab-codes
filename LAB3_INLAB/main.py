from functions.tabulate import tabulate
import heapq
from typing import List, Tuple, Optional

goal_state = [
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2]
]

def calculate_manhattan(state: List[List[int]]) -> int:
    return sum(abs(i - 3) + abs(j - 3) for i in range(7) for j in range(7) if state[i][j] == 1)

def calculate_exponential(state: List[List[int]]) -> int:
    return sum(2 ** max(abs(i - 3), abs(j - 3)) for i in range(7) for j in range(7) if state[i][j] == 1)

class State:
    def __init__(self, state: List[List[int]], parent: Optional['State'], action: Optional[List[List[int]]], heuristic: int, cost: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.cost = cost

    def __lt__(self, other: 'State') -> bool:
        return self.heuristic < other.heuristic

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
                        successors.append(State(new_state, None, [[i, j], [new_i, new_j]], heuristic, 0))

    return successors

def calculate_heuristic(state: List[List[int]], heuristic_type: str) -> int:
    return calculate_manhattan(state) if heuristic_type == "Manhattan" else calculate_exponential(state)

def construct_path(state: State) -> List[List[List[int]]]:
    path = []
    while state.parent:
        path.append(state.action)
        state = state.parent
    return list(reversed(path))

def a_star_search(start_state: List[List[int]], heuristic_type: str) -> Optional[Tuple[int, int]]:
    explored = set()
    frontier = []
    initial_state = State(start_state, None, None, calculate_heuristic(start_state, heuristic_type), 0)
    heapq.heappush(frontier, (initial_state.heuristic, 0, initial_state))

    while frontier:
        _, g, current_state = heapq.heappop(frontier)

        if is_goal(current_state.state):
            return len(construct_path(current_state)), len(explored)

        current_state_tuple = tuple(map(tuple, current_state.state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            for successor in get_successors(current_state.state, heuristic_type):
                successor.cost = g + 1
                f = successor.cost + successor.heuristic
                heapq.heappush(frontier, (f, successor.cost, successor))

    return None

def priority_queue_search(start_state: List[List[int]]) -> Optional[Tuple[int, int]]:
    explored = set()
    frontier = [(0, State(start_state, None, None, 0, 0))]

    while frontier:
        cost, current_state = heapq.heappop(frontier)

        if is_goal(current_state.state):
            return len(construct_path(current_state)), len(explored)

        current_state_tuple = tuple(map(tuple, current_state.state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            for successor in get_successors(current_state.state, "Manhattan"):
                successor.cost = cost + 1
                heapq.heappush(frontier, (successor.cost, successor))

    return None

def best_first_search(start_state: List[List[int]], heuristic_type: str) -> Optional[Tuple[int, int]]:
    explored = set()
    frontier = [(calculate_heuristic(start_state, heuristic_type), State(start_state, None, None, 0, 0))]

    while frontier:
        _, current_state = heapq.heappop(frontier)

        if is_goal(current_state.state):
            return len(construct_path(current_state)), len(explored)

        current_state_tuple = tuple(map(tuple, current_state.state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            for successor in get_successors(current_state.state, heuristic_type):
                heapq.heappush(frontier, (successor.heuristic, successor))

    return None

initial_state = [
    [2, 2, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 2, 2]
]

results = []
algorithms = [
    ("Best First Search", best_first_search, "Manhattan"),
    ("A* Search", a_star_search, "Manhattan"),
    ("Priority Queue Search", priority_queue_search, None)
]

for algorithm_name, algorithm_function, heuristic_type in algorithms:
    print(f"Executing {algorithm_name} with {heuristic_type if heuristic_type else 'N/A'} heuristic")
    solution = algorithm_function(initial_state, heuristic_type) if heuristic_type else algorithm_function(initial_state)

    if solution:
        results.append([algorithm_name, heuristic_type or 'N/A', "Solution found", solution[0], solution[1]])
    else:
        results.append([algorithm_name, heuristic_type or 'N/A', "No solution found", "N/A", "N/A"])

print(tabulate(results, headers=["Algorithm", "Heuristic", "Result", "Path Length", "Explored Set Length"]))
