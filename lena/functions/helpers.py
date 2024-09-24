import numpy as np
from PIL import Image

def load_file(file_path: str) -> np.ndarray:
    numbers = np.loadtxt(file_path)
    matrix = numbers.reshape(512, 512)
    return matrix

def find_similar_neighbors(grid, x, y, target_value, threshold):
    similar_neighbors = []
    neighbors = [
        (x-1, y), (x+1, y), (x, y-1), (x, y+1),  
        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  
    ]
    
    for nx, ny in neighbors:
        
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
            neighbor_value = grid[nx][ny]
        
            if abs(neighbor_value - target_value) <= threshold:
                similar_neighbors.append((nx, ny, neighbor_value))
    
    return similar_neighbors

def calculate_energy(grid, threshold):
    energy = 0

    for x in range(512):
        for y in range(512):
            current_value = grid[x][y]
            similar_neighbors = find_similar_neighbors(grid, x, y, current_value, threshold)
            energy += len(similar_neighbors) 

    return energy

def swap_pixels(grid):
    x1, y1 = np.random.randint(0, len(grid)), np.random.randint(0, len(grid[0]))
    x2, y2 = np.random.randint(0, len(grid)), np.random.randint(0, len(grid[0]))
    
    grid[x1][y1], grid[x2][y2] = grid[x2][y2], grid[x1][y1]
    return grid


def simulated_annealing(grid, initial_temp, cooling_rate, threshold, min_temp):
    iteration = 0
    current_grid = grid.copy()
    current_energy = calculate_energy(current_grid, threshold)
    best_grid = current_grid.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    
    while initial_temp > min_temp:
        new_grid = current_grid.copy()
        swap_pixels(new_grid)

        new_energy = calculate_energy(new_grid, threshold)
        """
            Acceptance Criteria : 
                - If the new energy is less than the current energy, accept the new grid
                - If the new energy is greater than the current energy, accept the new grid with a probability based on the temperature
        """

        if new_energy < current_energy:
            current_grid = new_grid
            current_energy = new_energy

        else:
            acceptance_probability = np.exp(-(new_energy - current_energy) / temperature)
            if np.random.rand() < acceptance_probability:
                current_grid = new_grid
                current_energy = new_energy

        if current_energy < best_energy:
            best_grid = current_grid.copy()
            best_energy = current_energy

        
        temperature *= cooling_rate
        iteration += 1

        if iteration % 100 == 0: 
            print(f"Iteration {iteration}, Current Energy: {current_energy}, Best Energy: {best_energy}")
            img = Image.fromarray(current_grid.astype(np.uint8), mode='L')
            img.save(f"reconstructed_image_{iteration}.png")

    return best_grid
