from collections import deque

def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if 3 - missionaries > 0 and 3 - missionaries < 3 - cannibals:
        return False
    return True

def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    if boat == 1:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    max_size = 0
    visited = set()
    while queue:
        (state, path) = queue.popleft()

        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            print('Total nodes explored from bfs', len(visited))
            print("MAX Queue size", max_size)
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
        if len(queue) > max_size:
            max_size = len(queue)
        # print('Total nodes explored from bfs', len(visited))
        # print("MAX Queue size", max_size)
    return None


def dfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    max_size = 0
    visited = set()
    while queue:
        (state, path) = queue.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            print('Total nodes explored from dfs', len(visited))
            print("MAX Stack size", max_size)
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
        if len(queue)>max_size:
            max_size=len(queue)
        # print('Total nodes explored from dfs', len(visited))
        # print("MAX Queue size", max_size)
    return None

start_state = (3, 3, 1)
goal_state = (0, 0, 0)

solution_bfs = bfs(start_state, goal_state)
count = 0
if solution_bfs:
    print("Solution found fom bfs:")
    for step in solution_bfs:
        count = count+1
        print(step)
    print("Solution nodes:", count)
else:
    print("No solution found.")


solution_dfs = dfs(start_state, goal_state)
count = 0
if solution_bfs:
    print("Solution found fom dfs:")
    for step in solution_dfs:
        count = count+1
        print(step)
    print("Solution nodes:", count)
else:
    print("No solution found.")
