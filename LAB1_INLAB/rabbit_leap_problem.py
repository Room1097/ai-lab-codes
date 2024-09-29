from collections import deque

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move


def get_successors(node):
    successors = []
    state = node.state
    empty_index = state.index('O')
    possible_moves = []


    if empty_index > 0 and state[empty_index-1]=='E':
        possible_moves.append(empty_index - 1)
    if empty_index < 6 and state[empty_index+1]=='W':
        possible_moves.append(empty_index + 1)


    if empty_index > 1 and state[empty_index-2]=='E':
        possible_moves.append(empty_index - 2)
    if empty_index < 5 and state[empty_index+2]=='W':
        possible_moves.append(empty_index + 2)

    for move in possible_moves:
        new_state = list(state)
        new_state[empty_index], new_state[move] = new_state[move], new_state[empty_index]
        successors.append(Node(new_state, node, move))

    return successors


def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    max_size = 0
    nodes_explored = 0
    while queue:
        node = queue.popleft()
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))

        # print(node.state)
        nodes_explored = nodes_explored + 1
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored from bfs', nodes_explored)
            print("MAX Queue size",max_size)
            return path[::-1]
        for successor in get_successors(node):
            queue.append(successor)
        if len(queue)>max_size:
            max_size=len(queue)

    print('Total nodes explored from bfs', nodes_explored)
    print("MAX Queue size", max_size)

    return None

def dfs(start_state, goal_state):
    start_node = Node(start_state)
    max_size = 0
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0
    while queue:
        node = queue.pop()
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        # print(node.state)
        nodes_explored = nodes_explored + 1
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored from dfs', nodes_explored)
            print("MAX Stack size",max_size)

            return path[::-1]
        for successor in get_successors(node):
            queue.append(successor)
        if len(queue)>max_size:
            max_size=len(queue)
    print('Total nodes explored from dfs', nodes_explored)
    print("MAX Stack size", max_size)
    return None


start_state = ('E', 'E', 'E', 'O', 'W', 'W', 'W')
goal_state = ('W', 'W', 'W', 'O', 'E', 'E', 'E')

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
