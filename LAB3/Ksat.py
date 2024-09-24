import random

# SAT Problem Generator
def generate_k_sat_problem(n, m, k):
    """
    Generates a random k-SAT problem.
    
    Args:
    - n: Number of variables
    - m: Number of clauses
    - k: Clause length (k-SAT)
    
    Returns:
    - A list of clauses representing the k-SAT formula.
      Each clause is a list of literals (variables or their negation).
    """
    problem = []
    for _ in range(m):
        clause = []
        variables = random.sample(range(1, n + 1), k)  # Pick k distinct variables
        for var in variables:
            if random.choice([True, False]):
                clause.append(var)  # Non-negated variable
            else:
                clause.append(-var)  # Negated variable
        problem.append(clause)
    
    return problem

# Heuristic Functions
def satisfied_clauses(formula, assignment):
    """Counts the number of clauses satisfied by the current assignment."""
    satisfied = 0
    for clause in formula:
        if any((lit > 0 and assignment[abs(lit) - 1] == 1) or (lit < 0 and assignment[abs(lit) - 1] == 0) for lit in clause):
            satisfied += 1
    return satisfied

def weighted_satisfied_clauses(formula, assignment):
    """Returns a weighted score of satisfied clauses, penalizing longer clauses."""
    weight_sum = 0
    for clause in formula:
        weight = 1 / len(clause)  # Weight inversely proportional to clause length
        if any((lit > 0 and assignment[abs(lit) - 1] == 1) or (lit < 0 and assignment[abs(lit) - 1] == 0) for lit in clause):
            weight_sum += weight
    return weight_sum

# Hill-Climbing Algorithm
def hill_climbing(formula, n, heuristic_func):
    """Solves SAT using Hill-Climbing."""
    current = [random.choice([0, 1]) for _ in range(n)]  # Initial random assignment
    current_score = heuristic_func(formula, current)
    
    while True:
        neighbors = []
        for i in range(n):
            neighbor = current.copy()
            neighbor[i] = 1 - neighbor[i]  # Flip one variable
            neighbors.append(neighbor)
        
        # Find the best neighbor
        neighbor_scores = [(neighbor, heuristic_func(formula, neighbor)) for neighbor in neighbors]
        best_neighbor, best_score = max(neighbor_scores, key=lambda x: x[1])
        
        if best_score <= current_score:  # No improvement
            break
        current = best_neighbor
        current_score = best_score
    
    return current, current_score

# Beam-Search Algorithm
def beam_search(formula, n, beam_width, heuristic_func):
    """Solves SAT using Beam Search."""
    current_beam = [[random.choice([0, 1]) for _ in range(n)] for _ in range(beam_width)]
    current_scores = [heuristic_func(formula, assignment) for assignment in current_beam]
    
    while True:
        all_neighbors = []
        for state in current_beam:
            for i in range(n):
                neighbor = state.copy()
                neighbor[i] = 1 - neighbor[i]  # Flip one variable
                all_neighbors.append(neighbor)
        
        neighbor_scores = [(neighbor, heuristic_func(formula, neighbor)) for neighbor in all_neighbors]
        best_neighbors = sorted(neighbor_scores, key=lambda x: -x[1])[:beam_width]
        
        if all(score <= min(current_scores) for _, score in best_neighbors):
            break
        
        current_beam = [neighbor for neighbor, _ in best_neighbors]
        current_scores = [score for _, score in best_neighbors]
    
    return current_beam[0], current_scores[0]

# Variable-Neighborhood-Descent (VND) Algorithm
def variable_neighborhood_descent(formula, n, heuristic_func):
    """Solves SAT using Variable Neighborhood Descent."""
    current = [random.choice([0, 1]) for _ in range(n)]  # Initial random assignment
    current_score = heuristic_func(formula, current)
    
    def neighborhood_1(state):
        return [state[:i] + [1 - state[i]] + state[i+1:] for i in range(n)]
    
    def neighborhood_2(state):
        return [state[:i] + [1 - state[i]] + state[i+1:j] + [1 - state[j]] + state[j+1:] for i in range(n) for j in range(i+1, n)]
    
    def neighborhood_3(state):
        return [state[:i] + [1 - state[i]] + state[i+1:j] + [1 - state[j]] + state[j+1:k] + [1 - state[k]] + state[k+1:]
                for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    neighborhoods = [neighborhood_1, neighborhood_2, neighborhood_3]
    
    for neighborhood in neighborhoods:
        while True:
            neighbors = neighborhood(current)
            neighbor_scores = [(neighbor, heuristic_func(formula, neighbor)) for neighbor in neighbors]
            best_neighbor, best_score = max(neighbor_scores, key=lambda x: x[1])
            
            if best_score <= current_score:  # No improvement
                break
            current = best_neighbor
            current_score = best_score
    
    return current, current_score

# Experiment Setup
m_values = [10, 20, 30]  # Different numbers of clauses
n_values = [5, 10, 15]  # Different numbers of variables
results = []

# Running experiments with different solvers
for m in m_values:
    for n in n_values:
        problem = generate_k_sat_problem(n, m, k=3)  # Generating a 3-SAT problem
        
        # Solving with Hill-Climbing
        hill_solution, hill_score = hill_climbing(problem, n, satisfied_clauses)
        
        # Solving with Beam Search (beam width 3)
        beam_solution, beam_score = beam_search(problem, n, beam_width=3, heuristic_func=satisfied_clauses)
        
        # Solving with Variable-Neighborhood-Descent
        vnd_solution, vnd_score = variable_neighborhood_descent(problem, n, satisfied_clauses)
        
        # Store results
        results.append({
            'm': m,
            'n': n,
            'hill_score': hill_score,
            'beam_score': beam_score,
            'vnd_score': vnd_score
        })

# Displaying results
for result in results:
    print(f"Clauses (m): {result['m']}, Variables (n): {result['n']}")
    print(f"  Hill-Climbing Score: {result['hill_score']}")
    print(f"  Beam Search Score (width 3): {result['beam_score']}")
    print(f"  VND Score: {result['vnd_score']}\n")
