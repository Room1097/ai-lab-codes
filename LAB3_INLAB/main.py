from functions.tabulate import tabulate
import heapq
from typing import List, Tuple, Optional

from functions.best_first_search import best_first_search
from functions.a_star import a_star_search


initial_state = [
    [2, 2, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 2, 2]
]

results = []
algorithms = [
    ("Best First Search", best_first_search, "Manhattan"),
    ("Best First Search", best_first_search, "Exponential"),
    ("A* Search", a_star_search, "Manhattan"),
    ("A* Search", a_star_search, "Exponential")
]

for algorithm_name, algorithm_function, heuristic_type in algorithms:
    print(f"Executing {algorithm_name} with {heuristic_type} heuristic")
    solution = algorithm_function(initial_state, heuristic_type)

    if solution:
        results.append([algorithm_name, heuristic_type, "Solution found", solution[0], solution[1]])
    else:
        results.append([algorithm_name, heuristic_type, "No solution found", "N/A", "N/A"])

print(tabulate(results, headers=["Algorithm", "Heuristic", "Result", "Path Length", "Explored Set Length"]))
