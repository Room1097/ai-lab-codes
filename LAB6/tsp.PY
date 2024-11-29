import numpy as np

# Parameters
N = 10  # Number of cities
A, B, C = 500, 500, 200  # Weights for energy terms
T = 1.0  # Temperature parameter (controls randomness)
iterations = 2000  # Number of iterations

# Random distance matrix (D)
np.random.seed(0)  # For reproducibility
D = np.random.randint(1, 100, size=(N, N))  # Random distances
np.fill_diagonal(D, 0)  # Set diagonal to 0 (no self-loop)

# Initialize neuron states (random values between 0 and 1)
x = np.random.rand(N, N)

# Hopfield network update function
def hopfield_update(x, D, A, B, C, T, iterations):
    for _ in range(iterations):
        # Compute input to neurons
        u = (
            -A * (np.sum(x, axis=1, keepdims=True) - 1)  # Constraint 1: Each city is visited once
            -B * (np.sum(x, axis=0, keepdims=True) - 1)  # Constraint 2: Each position is occupied once
            -C * np.roll(np.dot(D, x), shift=-1, axis=1)  # Minimize travel distance
        )
        # Update states using sigmoid activation function
        x = 1 / (1 + np.exp(-u / T))
    return x

# Run the Hopfield network
x_final = hopfield_update(x, D, A, B, C, T, iterations)

# Decode the solution
solution = np.argmax(x_final, axis=1)

# Ensure solution is valid
def validate_solution(solution):
    unique_cities = len(set(solution))
    return unique_cities == N  # Each city must be visited exactly once

# Print results
print("Distance Matrix (D):\n", D)
print("\nNeuron Activations (x_final):\n", np.round(x_final, 2))
print("\nTour (decoded):", solution)

if validate_solution(solution):
    print("\nThe solution is valid!")
else:
    print("\nThe solution is invalid. Adjust parameters or iterations.")
