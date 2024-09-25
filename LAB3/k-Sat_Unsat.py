import random

# Generate Random 3-SAT Problem
def generate_3sat_problem(n, m):
    clauses = []
    for _ in range(m):
        clause = random.sample(range(1, n + 1), 3)  
        clause = [v if random.choice([True, False]) else -v for v in clause]  
        clauses.append(clause)
    return clauses


def unsatisfied_clauses(clauses, assignment):
    """Returns the number of unsatisfied clauses."""
    unsatisfied = 0
    for clause in clauses:
        if not any((lit > 0 and assignment[lit - 1]) or (lit < 0 and not assignment[-lit - 1]) for lit in clause):
            unsatisfied += 1
    return unsatisfied



def hill_climbing_unsat(clauses, n, max_steps=1000):
    assignment = [random.choice([True, False]) for _ in range(n)]
    for _ in range(max_steps):
        current_unsat = unsatisfied_clauses(clauses, assignment)
        if current_unsat == 0:
            return assignment, True  # Solution found
        best_flip = None
        best_unsat = current_unsat
        for i in range(n):
            assignment[i] = not assignment[i]  
            new_unsat = unsatisfied_clauses(clauses, assignment)
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_flip = i
            assignment[i] = not assignment[i] 
        if best_flip is None:  
            break
        assignment[best_flip] = not assignment[best_flip]  
    return assignment, False  # No solution found


def beam_search_unsat(clauses, n, beam_width=3, max_steps=1000):
    beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
    best_assignment = None
    best_unsat = float('inf')
    
    for _ in range(max_steps):
        new_beam = []
        for assignment in beam:
            current_unsat = unsatisfied_clauses(clauses, assignment)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment
            if current_unsat == 0:
                return assignment, True  # Solution found
            for i in range(n):
                new_assignment = assignment[:]
                new_assignment[i] = not new_assignment[i]
                new_beam.append((new_assignment, unsatisfied_clauses(clauses, new_assignment)))
        # Sort and keep top k beam_width
        new_beam = sorted(new_beam, key=lambda x: x[1])[:beam_width]
        beam = [b[0] for b in new_beam]
    
    return best_assignment, False  # Return the best assignment seen so far


def vnd_unsat(clauses, n, max_steps=1000):
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
    for _ in range(max_steps):
        current_unsat = unsatisfied_clauses(clauses, assignment)
        if current_unsat == 0:
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
                new_unsat = unsatisfied_clauses(clauses, new_assignment)
                if new_unsat < current_unsat:
                    assignment = new_assignment
                    current_unsat = new_unsat
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return assignment, False


def experiment(n, m, trials=10):
    """Runs each algorithm and compares them based on success rate and average unsatisfied clauses."""
    results = {
        'hill_climbing': {'success': 0, 'average_unsat': 0},
        'beam_search_3': {'success': 0, 'average_unsat': 0},
        'beam_search_4': {'success': 0, 'average_unsat': 0},
        'vnd': {'success': 0, 'average_unsat': 0},
    }

    for _ in range(trials):
        clauses = generate_3sat_problem(n, m)

        # Hill-Climbing
        assignment, success = hill_climbing_unsat(clauses, n)
        results['hill_climbing']['success'] += int(success)
        results['hill_climbing']['average_unsat'] += unsatisfied_clauses(clauses, assignment)
        # Beam Search (beam width 3)
        assignment, success = beam_search_unsat(clauses, n, beam_width=3)
        results['beam_search_3']['success'] += int(success)
        results['beam_search_3']['average_unsat'] += unsatisfied_clauses(clauses, assignment)
        # Beam Search (beam width 4)
        assignment, success = beam_search_unsat(clauses, n, beam_width=4)
        results['beam_search_4']['success'] += int(success)
        results['beam_search_4']['average_unsat'] += unsatisfied_clauses(clauses, assignment)
        # Variable Neighborhood Descent
        assignment, success = vnd_unsat(clauses, n)
        results['vnd']['success'] += int(success)
        results['vnd']['average_unsat'] += unsatisfied_clauses(clauses, assignment)

    for alg in results:
        results[alg]['average_unsat'] /= trials

    return results

#  n variables and m clauses
n = 20
m = 80
trials = 5
results = experiment(n, m, trials)

print ("n =",n,"m =",m)
print("Comparison Results for 3-SAT Problem(Unsatisfied Clauses Hueristic):")
print(f"Hill-Climbing: Success Rate: {results['hill_climbing']['success']}/{trials}, Avg. Unsatisfied Clauses: {results['hill_climbing']['average_unsat']:.2f}")
print(f"Beam Search(width: 3): Success Rate: {results['beam_search_3']['success']}/{trials}, Avg. Unsatisfied Clauses: {results['beam_search_3']['average_unsat']:.2f}")
print(f"Beam Search(width: 4): Success Rate: {results['beam_search_4']['success']}/{trials}, Avg. Unsatisfied Clauses: {results['beam_search_4']['average_unsat']:.2f}")
print(f"VND: Success Rate: {results['vnd']['success']}/{trials}, Avg. Unsatisfied Clauses: {results['vnd']['average_unsat']:.2f}")
