import heapq
from .helpers import is_goal, get_successors, construct_path, calculate_heuristic

def a_star_search(start_state, heuristic_type):
    explored = set()
    frontier = []
    initial_heuristic = calculate_heuristic(start_state, heuristic_type)
    
    # Push the initial state onto the frontier
    heapq.heappush(frontier, (initial_heuristic, 0, start_state, None, None))  # (f, g, state, parent, action)

    while frontier:
        f, g, current_state, parent, action = heapq.heappop(frontier)

        if is_goal(current_state):
            path = construct_path((current_state, parent, action))
            return len(explored), len(path)

        # Convert current state to tuple for the explored set
        current_state_tuple = tuple(tuple(row) for row in current_state)

        if current_state_tuple not in explored:
            explored.add(current_state_tuple)

            # Get successors for the current state
            successors = get_successors(current_state, heuristic_type)

            for child_node in successors:
                child_state = child_node.state
                child_action = child_node.action
                child_heuristic = child_node.heuristic
                child_g = g + 1  # Increment cost for each move
                f = child_g + child_heuristic  # f = g + h

                child_state_tuple = tuple(tuple(row) for row in child_state)

                # Only push if the child has not been explored or if it's better than the current one in the frontier
                if child_state_tuple not in explored:
                    heapq.heappush(frontier, (f, child_g, child_state, current_state, child_action))

    return None
