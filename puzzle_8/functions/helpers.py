import random

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

def is_goal(state, goal):
    return state == goal

def possible_action(state):
    ind = state.index('_')
    moves = {
        0: [1, 3],                  # Top-left corner
        1: [-1, 1, 3],              # Top-middle
        2: [-1, 3],                 # Top-right corner
        3: [-3, 1, 3],              # Middle-left
        4: [-3, -1, 1, 3],          # Center
        5: [-3, -1, 3],             # Middle-right
        6: [-3, 1],                 # Bottom-left corner
        7: [-3, -1, 1],             # Bottom-middle
        8: [-3, -1],                # Bottom-right corner
    }
    
    return moves[ind]

def make_action(state, action):
    i = state.index('_')
    new_state = state[:]

    new_state[i], new_state[i + action] = new_state[i + action], new_state[i]
    
    return new_state

def backtrack_path(state: Node):
    path = []
    while state.parent is not None:
        path.append(state.state)
        state = state.parent
    return path[::-1]


def generate_random_state():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, '_']
    random.shuffle(numbers)
    return numbers
