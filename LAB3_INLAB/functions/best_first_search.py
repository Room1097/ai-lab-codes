import heapq
from typing import List, Optional, Tuple

from .helpers import State, calculate_heuristic, get_successors, is_goal, construct_path


def best_first_search(start_state: List[List[int]], heuristic_type: str) -> Optional[Tuple[int, int]]:
    explored = set()
    frontier = []
    initial_state = State(start_state, None, None, calculate_heuristic(start_state, heuristic_type), 0)
    heapq.heappush(frontier, (initial_state.heuristic, initial_state))

    while frontier:
        _, current_state = heapq.heappop(frontier)

        if is_goal(current_state.state):
            return len(construct_path(current_state)), len(explored)

        current_state_tuple = tuple(map(tuple, current_state.state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            for successor in get_successors(current_state.state, heuristic_type):
                successor.parent = current_state
                heapq.heappush(frontier, (successor.heuristic, successor))

    return None
