import random
import math
import matplotlib.pyplot as plt

# Read the locations from a file
def load_locations(file_path):
    locations = {}
    with open(file_path, 'r') as file:
        for line in file:
            city, lat, lon = line.strip().split(',')
            locations[city] = (float(lat), float(lon))
    return locations

# Calculate Euclidean distance between two locations
def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Calculate total distance of a given route
def total_distance(route, locations):
    return sum(distance(locations[route[i]], locations[route[i + 1]]) for i in range(len(route) - 1)) + distance(locations[route[-1]], locations[route[0]])

# Simulated Annealing algorithm
def simulated_annealing(locations, initial_route, initial_temp, cooling_rate, max_iter):
    current_route = initial_route[:]
    current_cost = total_distance(current_route, locations)
    best_route = current_route[:]
    best_cost = current_cost
    temperature = initial_temp

    for iteration in range(max_iter):
        # Generate neighbor by swapping two cities
        new_route = current_route[:]
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        # Calculate new cost
        new_cost = total_distance(new_route, locations)
        
        # Acceptance probability
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_route = new_route
            current_cost = new_cost
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

        # Reduce temperature
        temperature *= cooling_rate

        # Print progress every 1000 iterations
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Best Cost = {best_cost:.4f}")

    return best_route, best_cost

# Load the locations from file
locations_file = 'LAB4\\rajasthan_locations.txt'
locations = load_locations(locations_file)

# Initial parameters
initial_route = list(locations.keys())
initial_temp = 10000
cooling_rate = 0.9
max_iter = 50000

# Perform Simulated Annealing
best_route, best_cost = simulated_annealing(locations, initial_route, initial_temp, cooling_rate, max_iter)
print(f"Optimal Route: {' -> '.join(best_route)}")
print(f"Minimum Distance: {best_cost:.4f}")

# Visualize the route
def plot_route(route, locations):
    plt.figure(figsize=(10, 6))
    x, y = zip(*[locations[city] for city in route])
    plt.plot(x + (x[0],), y + (y[0],), 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city)
    plt.title(f'Optimal Tour - Total Distance: {best_cost:.4f}')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.grid()
    plt.show()

plot_route(best_route, locations)
