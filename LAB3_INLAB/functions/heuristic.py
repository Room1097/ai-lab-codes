from typing import List

def calculate_manhattan(state: List[List[int]]) -> int:
    return sum(abs(i - 3) + abs(j - 3) for i in range(7) for j in range(7) if state[i][j] == 1)

def calculate_exponential(state: List[List[int]]) -> int:
    return sum(2 ** max(abs(i - 3), abs(j - 3)) for i in range(7) for j in range(7) if state[i][j] == 1)

