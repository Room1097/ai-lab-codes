import numpy as np
import matplotlib.pyplot as plt

class HopfieldTSP:
    def __init__(self, cities, beta=10, alpha=0.1):
        """
        Initialize Hopfield Network for TSP
        
        Args:
        cities: 2D array of city coordinates
        beta: Temperature parameter
        alpha: Learning rate
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.beta = beta
        self.alpha = alpha
        
        # Distance matrix between cities
        self.distance_matrix = self.calculate_distance_matrix()
        
        # Initialize network weights
        self.weights = np.zeros((self.num_cities, self.num_cities, self.num_cities, self.num_cities))
        self.initialize_weights()
        
        # Initialize state matrix
        self.state = np.zeros((self.num_cities, self.num_cities))
    
    def calculate_distance_matrix(self):
        """Calculate Euclidean distance between cities"""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                dist_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist_matrix
    
    def initialize_weights(self):
        """Initialize Hopfield network weights"""
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                for k in range(self.num_cities):
                    for l in range(self.num_cities):
                        # Constraints weights
                        w1 = -1  # Ensure one city per position
                        w2 = -1  # Ensure one position per city
                        w3 = self.distance_matrix[i][j]  # Distance penalty
                        
                        self.weights[i][j][k][l] = self.alpha * (w1 + w2 + w3)
    
    def update_state(self):
        """Update network state using sigmoid activation"""
        new_state = np.zeros_like(self.state)
        
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                # Calculate local field
                local_field = 0
                for k in range(self.num_cities):
                    for l in range(self.num_cities):
                        local_field += self.weights[i][j][k][l] * self.state[k][l]
                
                # Sigmoid activation
                prob = 1 / (1 + np.exp(-self.beta * local_field))
                new_state[i][j] = 1 if prob > np.random.random() else 0
        
        self.state = new_state
    
    def solve_tsp(self, max_iterations=1000):
        """Solve TSP using Hopfield Network"""
        # Random initial state
        self.state = np.random.randint(2, size=(self.num_cities, self.num_cities))
        
        for _ in range(max_iterations):
            prev_state = self.state.copy()
            self.update_state()
            
            # Check convergence
            if np.array_equal(prev_state, self.state):
                break
        
        # Extract tour from final state
        tour = self.extract_tour()
        return tour
    
    def extract_tour(self):
        """Extract optimal tour from final state"""
        tour = []
        for j in range(self.num_cities):
            for i in range(self.num_cities):
                if self.state[i][j] == 1:
                    tour.append(i)
                    break
        return tour
    
    def calculate_tour_length(self, tour):
        """Calculate total tour length"""
        total_length = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_length += self.distance_matrix[current_city][next_city]
        return total_length

def generate_random_cities(num_cities):
    """Generate random city coordinates"""
    return np.random.rand(num_cities, 2) * 100

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 10 random cities
    num_cities = 10
    cities = generate_random_cities(num_cities)
    
    # Solve TSP using Hopfield Network
    hopfield_tsp = HopfieldTSP(cities)
    tour = hopfield_tsp.solve_tsp()
    tour_length = hopfield_tsp.calculate_tour_length(tour)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100)
    
    # Plot tour
    for i in range(len(tour)):
        start = cities[tour[i]]
        end = cities[tour[(i + 1) % len(tour)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')
    
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), xytext=(5, 5), 
                     textcoords='offset points')
    
    plt.title(f'TSP Tour (Length: {tour_length:.2f})')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()
    
    print("Optimal Tour:", tour)
    print(f"Tour Length: {tour_length:.2f}")

if __name__ == "__main__":
    main()