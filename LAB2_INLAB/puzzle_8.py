from collections import deque
import time
import math
import random

# Node definition
class Node:
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.depth = depth  # Keep track of depth for depth-limited search

# Generate successors
def get_successors(node):
    successors = []
    index = node.state.index(0)
    moves = [-1, 1, 3, -3]
    for move in moves:
        im = index + move
        # Check for valid move within boundaries
        if im >= 0 and im < 9 and not (index % 3 == 0 and move == -1) and not (index % 3 == 2 and move == 1):
            new_state = list(node.state)
            new_state[index], new_state[im] = new_state[im], new_state[index]
            successor = Node(new_state, node, node.depth + 1)
            successors.append(successor)            
    return successors

# Depth-Limited Search (DLS)
def dls(node, goal_state, limit):
    if node.state == goal_state:
        return node  # Goal found
    elif node.depth == limit:
        return None  # Depth limit reached

    for successor in get_successors(node):
        result = dls(successor, goal_state, limit)
        if result:
            return result
    return None

# Iterative Deepening Search (IDS)
def ids(start_state, goal_state):
    depth = 0
    while True:
        print(f"Trying depth limit: {depth}")
        result = dls(Node(start_state), goal_state, depth)
        if result:
            return result  # Return goal node if found
        depth += 1  # Increase depth limit

# Backtrack and generate path from goal node
def get_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# Generate random Puzzle-8 state at depth 'd' from the start state
def generate_puzzle8_at_depth(d):
    start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    s_node = Node(start_state)
    for _ in range(d):
        s_node = random.choice(get_successors(s_node))
    return s_node.state

# Test the IDS with different depth levels
def get_path_for_depth(depth):
    start_state = [1, 2, 3, 4, 5, 6, 7, 0, 8]
    goal_state = generate_puzzle8_at_depth(depth)
    print(f"Generated goal state at depth {depth}: {goal_state}")
    
    start_time = time.time()
    
    result = ids(start_state, goal_state)
    if result:
        path = get_path(result)
        print("Solution found by IDS:")
        for step in path:
            print(step)
        print(f"Total nodes in solution path: {len(path)}")
    else:
        print("No solution found.")
    
    end_time = time.time()
    print("Time taken:", math.ceil((end_time - start_time) * 1000), "ms")

get_path_for_depth(10)
get_path_for_depth(20)
get_path_for_depth(30)
get_path_for_depth(40)
get_path_for_depth(50)
get_path_for_depth(100)


