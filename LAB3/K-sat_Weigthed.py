import random

# Generate Random 3-SAT Problem
def generate_3sat_problem(n, m):

    clauses = []
    for _ in range(m):
        clause = random.sample(range(1, n + 1), 3)  # Select 3 distinct variables
        clause = [v if random.choice([True, False]) else -v for v in clause]  # Random negation
        clauses.append(clause)
    return clauses


def clause_weighting_heuristic(clauses, assignment, clause_weights):

    unsat_clauses_weight = 0
    for i, clause in enumerate(clauses):
        if not any((lit > 0 and assignment[lit - 1]) or (lit < 0 and not assignment[-lit - 1]) for lit in clause):
            unsat_clauses_weight += clause_weights[i]
    return unsat_clauses_weight

# Update clause weights based on unsatisfied clauses
def update_clause_weights(clauses, assignment, clause_weights):

    for i, clause in enumerate(clauses):
        if not any((lit > 0 and assignment[lit - 1]) or (lit < 0 and not assignment[-lit - 1]) for lit in clause):
            clause_weights[i] += 1


def hill_climbing_clause_weighting(clauses, n, max_steps=1000, restarts=10):

    best_assignment = None
    best_score = float('inf')
    clause_weights = [1] * len(clauses) 

    for restart in range(restarts):
        assignment = [random.choice([True, False]) for _ in range(n)]
        for step in range(max_steps):
            current_score = clause_weighting_heuristic(clauses, assignment, clause_weights)
            if current_score == 0:
                return assignment, True  # Solution found
            
            best_flip = None
            best_new_score = current_score
            for i in range(n):
                assignment[i] = not assignment[i]  
                new_score = clause_weighting_heuristic(clauses, assignment, clause_weights)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_flip = i
                assignment[i] = not assignment[i]  
            if best_flip is None:  
                break
            assignment[best_flip] = not assignment[best_flip] 
            update_clause_weights(clauses, assignment, clause_weights) 

        if current_score < best_score:
            best_score = current_score
            best_assignment = assignment

    return best_assignment, best_score == 0  


def beam_search_clause_weighting(clauses, n, beam_width=5, max_steps=1000):

    beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
    best_assignment = None
    best_score = float('inf')
    clause_weights = [1] * len(clauses)  

    for _ in range(max_steps):
        new_beam = []
        for assignment in beam:
            current_score = clause_weighting_heuristic(clauses, assignment, clause_weights)
            if current_score < best_score:
                best_score = current_score
                best_assignment = assignment
            if current_score == 0:
                return assignment, True  # Solution found
            
            for i in range(n):
                new_assignment = assignment[:]
                new_assignment[i] = not new_assignment[i]
                new_beam.append((new_assignment, clause_weighting_heuristic(clauses, new_assignment, clause_weights)))

        new_beam = sorted(new_beam, key=lambda x: x[1])[:beam_width]
        beam = [b[0] for b in new_beam]
        update_clause_weights(clauses, beam[0], clause_weights)  

    return best_assignment, best_score == 0  


def vnd_clause_weighting(clauses, n, max_steps=1000):

    
    def neighborhood_1(assignment):

        for i in range(n):
            yield i

    def neighborhood_2(assignment):

        for i in range(n):
            for j in range(i + 1, n):
                yield i, j

    def neighborhood_3(assignment):

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    yield i, j, k

    neighborhoods = [neighborhood_1, neighborhood_2, neighborhood_3]
    assignment = [random.choice([True, False]) for _ in range(n)]
    clause_weights = [1] * len(clauses)  
    for step in range(max_steps):
        current_score = clause_weighting_heuristic(clauses, assignment, clause_weights)
        if current_score == 0:
            return assignment, True  # Solution found
        improved = False
        for neighborhood in neighborhoods:
            for flip_vars in neighborhood(assignment):
                new_assignment = assignment[:]
                if isinstance(flip_vars, tuple):
                    for var in flip_vars:
                        new_assignment[var] = not new_assignment[var]
                else:
                    new_assignment[flip_vars] = not new_assignment[flip_vars]
                new_score = clause_weighting_heuristic(clauses, new_assignment, clause_weights)
                if new_score < current_score:
                    assignment = new_assignment
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        update_clause_weights(clauses, assignment, clause_weights)  # Update weights

    return assignment, current_score == 0


def experiment(n, m, trials=5):
    """Runs each algorithm using the clause weighting heuristic and compares them based on success rate and average clause weight."""
    results = {
        'hill_climbing': {'success': 0, 'average_weighted_unsat': 0},
        'beam_search_3': {'success': 0, 'average_weighted_unsat': 0},
        'beam_search_4': {'success': 0, 'average_weighted_unsat': 0},
        'vnd': {'success': 0, 'average_weighted_unsat': 0},
    }

    for _ in range(trials):
        clauses = generate_3sat_problem(n, m)
        clause_weights = [1] * m  # Initialize clause weights to 1 for each clause

        # Hill-Climbing
        assignment, success = hill_climbing_clause_weighting(clauses, n)
        results['hill_climbing']['success'] += int(success)
        results['hill_climbing']['average_weighted_unsat'] += clause_weighting_heuristic(clauses, assignment, clause_weights)

        # Beam Search (beam width 3)
        assignment, success = beam_search_clause_weighting(clauses, n, beam_width=3)
        results['beam_search_3']['success'] += int(success)
        results['beam_search_3']['average_weighted_unsat'] += clause_weighting_heuristic(clauses, assignment, clause_weights)

        # Beam Search (beam width 4)
        assignment, success = beam_search_clause_weighting(clauses, n, beam_width=4)
        results['beam_search_4']['success'] += int(success)
        results['beam_search_4']['average_weighted_unsat'] += clause_weighting_heuristic(clauses, assignment, clause_weights)

        # Variable Neighborhood Descent (VND)
        assignment, success = vnd_clause_weighting(clauses, n)
        results['vnd']['success'] += int(success)
        results['vnd']['average_weighted_unsat'] += clause_weighting_heuristic(clauses, assignment, clause_weights)


    for alg in results:
        results[alg]['average_weighted_unsat'] /= trials

    return results

# n variables and m clauses
n = 20
m = 80
trials = 5
results = experiment(n, m, trials)

print ("n =",n,"m =",m)
print("Comparison Results for 3-SAT Problem (Clause Weighting Heuristic):")
print(f"Hill-Climbing: Success Rate: {results['hill_climbing']['success']}/{trials}, Avg. Weighted Unsatisfied Clauses: {results['hill_climbing']['average_weighted_unsat']:.2f}")
print(f"Beam Search(width: 3): Success Rate: {results['beam_search_3']['success']}/{trials}, Avg. Weighted Unsatisfied Clauses: {results['beam_search_3']['average_weighted_unsat']:.2f}")
print(f"Beam Search(width: 4): Success Rate: {results['beam_search_4']['success']}/{trials}, Avg. Weighted Unsatisfied Clauses: {results['beam_search_4']['average_weighted_unsat']:.2f}")
print(f"VND: Success Rate: {results['vnd']['success']}/{trials}, Avg. Weighted Unsatisfied Clauses: {results['vnd']['average_weighted_unsat']:.2f}")
