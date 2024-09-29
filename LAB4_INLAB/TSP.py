import random
import math
import matplotlib.pyplot as plt

def parse_tsp_file(file_path):
    coordinates = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        node_section = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                node_section = True
                continue
            if line.strip() == "EOF":
                break
            if node_section:
                parts = line.strip().split()
                index = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates[index] = (x, y)
    return coordinates

# Calculate Euclidean distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(route, coordinates):
    return sum(distance(coordinates[route[i]], coordinates[route[i + 1]]) for i in range(len(route) - 1)) + distance(coordinates[route[-1]], coordinates[route[0]])

def simulated_annealing(initial_route, coordinates, initial_temp, cooling_rate, max_iter):
    current_route = initial_route[:]
    current_cost = total_distance(current_route, coordinates)
    best_route = current_route[:]
    best_cost = current_cost
    temperature = initial_temp

    for iteration in range(max_iter):
        new_route = current_route[:]
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_cost = total_distance(new_route, coordinates)
        
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_route = new_route
            current_cost = new_cost
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

        temperature *= cooling_rate

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Best Cost = {best_cost:.4f}")

    return best_route, best_cost

def plot_route(route, coordinates, best_cost):
    plt.figure(figsize=(10, 6))
    x, y = zip(*[coordinates[city] for city in route])
    plt.plot(x + (x[0],), y + (y[0],), 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], str(city))
    plt.title(f'Optimal Tour - Total Distance: {best_cost:.4f}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.show()

def main():
    file_path = 'LAB4\\bcl380.tsp'
    coordinates = parse_tsp_file(file_path)
    
    
    initial_route = list(coordinates.keys())
    # random.shuffle(initial_route)
    # initial_route.remove(1)  # Remove city 1 from the initial list
    # random.shuffle(initial_route)  # Shuffle the remaining cities
    # initial_route = [1] + initial_route  # Ensure the route starts with city 1
    initial_temp = 10000
    cooling_rate = 0.80
    max_iter = 1000000
    

    best_route, best_cost = simulated_annealing(initial_route, coordinates, initial_temp, cooling_rate, max_iter)
    print(f"Optimal Route: {' -> '.join(map(str, best_route))}")
    print(f"Minimum Distance: {best_cost:.4f}")
    
    plot_route(best_route, coordinates, best_cost)

main()
