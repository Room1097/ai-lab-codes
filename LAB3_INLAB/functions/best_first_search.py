import heapq
from .helpers import is_goal, get_successors, construct_path, calculate_heuristic

def best_first_search(start_state, heuristic_type):
    explored = set()
    frontier = []
    initial_heuristic = calculate_heuristic(start_state, heuristic_type)
    
    # Create an initial node that maintains references
    initial_node = (start_state, None, None)  # (state, parent, action)
    heapq.heappush(frontier, (initial_heuristic, initial_node))

    while frontier:
        _, current_node = heapq.heappop(frontier)
        current_state, parent, action = current_node

        # Convert state to tuple for comparison in explored
        current_state_tuple = tuple(tuple(row) for row in current_state)

        if is_goal(current_state):

            path = construct_path(current_node)
            return len(path), len(explored)
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            successors = get_successors(current_state, heuristic_type)

            for child_node in successors:
                child_state = child_node.state
                child_action = child_node.action
                child_heuristic = child_node.heuristic

                
                child_state_tuple = tuple(tuple(row) for row in child_state)

                if child_state_tuple not in explored:
                    # Add child node to frontier based on its heuristic
                    heapq.heappush(frontier, (child_heuristic, (child_state, current_node, child_action)))

    return None
