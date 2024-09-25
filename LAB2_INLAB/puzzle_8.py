from functions import helpers
from functions import graph_search
from functions import decorators


# initial = [1, 2, 3, 4, 5, 6, 7, 8, '_']
# initial_node = helpers.Node(initial)
# goal = helpers.generate_random_state()

# initial = helpers.generate_random_state()
# initial_node = helpers.Node(initial)
# goal = [1, 2, 3, 4, 5, 6, 7, 8, '_']

initial = [1, 2, 3, '_',4, 6, 7, 5, 8]
initial_node = helpers.Node(initial)
goal = [1, 2, 3, 4, 5, 6, 7, 8, '_']

bfs_solution_path = graph_search.bfs(initial_node, goal)
dfs_solution_path = graph_search.dfs(initial_node, goal)
# print(solution_path)

print("Intial State:\n", initial)
print("Goal State:\n", goal)

decorators.bfs(bfs_solution_path)
decorators.dfs(dfs_solution_path)