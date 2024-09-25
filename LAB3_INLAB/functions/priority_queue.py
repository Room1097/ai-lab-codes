import heapq
from .helpers import is_goal, get_successors, construct_path

def priority_queue_search(start_state):
    explored = set()
    frontier = []
    
    # Initialize the frontier with the start state
    heapq.heappush(frontier, (0, start_state, None, None))  # (cost, state, parent, action)

    while frontier:
        cost, current_state, parent, action = heapq.heappop(frontier)

        # Check if the current state is the goal
        if is_goal(current_state):
            path = construct_path((current_state, parent, action))
            return len(explored), len(path)

        # Convert current state to tuple for the explored set
        current_state_tuple = tuple(tuple(row) for row in current_state)

        if current_state_tuple not in explored:
            explored.add(current_state_tuple)

            # Get successors for the current state
            successors = get_successors(current_state, "Manhattan")

            for child_node in successors:
                child_state = child_node.state
                child_action = child_node.action
                child_cost = cost + 1  # Increment the cost for each move

                child_state_tuple = tuple(tuple(row) for row in child_state)

                # Only add if it has not been explored
                if child_state_tuple not in explored:
                    heapq.heappush(frontier, (child_cost, child_state, current_state, child_action))

    return None
