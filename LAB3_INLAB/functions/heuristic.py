def calculate_manhattan(state):
    res = 0
    for i in range(7):
        for j in range(7):
            if state[i][j]==1:
                res += abs(3-i) + abs(3-j)

    return res

def calculate_exponential(state):
    res = 0
    for i in range(7):
        for j in range(7):
            if state[i][j]==1:
                res += 2 ** (max(abs(3 - i), abs(3 - j))) 
    return res