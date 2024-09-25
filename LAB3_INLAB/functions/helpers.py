from .heuristic import calculate_manhattan, calculate_exponential

class State:
    def __init__(self, state, parent, action, heuristic, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.cost = cost

    def __lt__(self, other):
        return self.heuristic < other.heuristic

goal_state = [
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2]
]

def is_goal(state):
    return state == goal_state

def get_successors(state, heuristic_type):
    successors = []
    dx = [[0, 0, 1, -1], [0, 0, 2, -2]]  
    dy = [[-1, 1, 0, 0], [-2, 2, 0, 0]]  

    for i in range(7):
        for j in range(7):
            if state[i][j] == 1:  
                for k in range(4):
                
                    new_i = i + dy[1][k]
                    new_j = j + dx[1][k]
                
                    mid_i = i + dy[0][k]
                    mid_j = j + dx[0][k]

                
                    if new_i < 0 or new_i >= 7 or new_j < 0 or new_j >= 7:
                        continue
                
                    if state[mid_i][mid_j] == 1 and state[new_i][new_j] == 0:
                        
                        stateCpy = [row.copy() for row in state]
                        stateCpy[new_i][new_j] = 1 
                        stateCpy[mid_i][mid_j] = 0 
                        stateCpy[i][j] = 0 
                        
                        if heuristic_type == "Manhattan":
                            heuristic = calculate_manhattan(stateCpy)
                        else:
                            heuristic = calculate_exponential(stateCpy)
                        
                        child = State(stateCpy, state, [[i, j], [new_i, new_j]], heuristic, 1)
                        successors.append(child)
    return successors


def construct_path(curr):
    print("Calculating path")
    path = []
    
    
    while curr[1] is not None:  
        print(f"Current action: {curr[2]}, Path length: {len(path)}")  # Debug print
        path.append(curr[2])  
        curr = (curr[1], curr[1][1], None)  

    path.reverse()  
    print("Path calculated:", path)
    return path



def calculate_heuristic(state, heuristic_type):
    if heuristic_type == "Manhattan":
        return calculate_manhattan(state)
    elif heuristic_type == "Exponential":
        return calculate_exponential(state)
    return 0