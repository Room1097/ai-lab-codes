def bfs(bfs_solution_path):
    if bfs_solution_path:
        print("BFS Solution Path:")
        for state in bfs_solution_path:
            print(state)
    else:
        print('No Solution Could be Obtained!')

def dfs(dfs_solution_path):
    if dfs_solution_path:
        print("DFS Solution Path:")
        for state in dfs_solution_path:
            print(state)
    else:
        print('No Solution Could be Obtained!')