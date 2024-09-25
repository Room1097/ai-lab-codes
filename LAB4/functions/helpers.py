import numpy as np
import random

def load_file(file_path):
    numbers = np.loadtxt(file_path)
    matrix = numbers.reshape(512, 512)
    matrix = matrix.transpose()
    return matrix

def calculate_energy(grid):
    energy = 0
    for i in range(512):
        for j in range(512):
            if j < 511:  
                if (j + 1) % 128 == 0:
                    energy += abs(grid[i, j] - grid[i, j + 1])
            if i < 511:  
                if (i + 1) % 128 == 0:
                    energy += abs(grid[i, j] - grid[i + 1, j])
    return energy

def swap_pieces(grid):
    i, j = random.sample(range(16), 2)
    r1, r2 = i // 4, j // 4
    c1, c2 = i % 4, j % 4
    
    rn1, rn2 = 128 * r1, 128 * r2
    cn1, cn2 = 128 * c1, 128 * c2
    
    piece1 = grid[rn1:rn1 + 128, cn1:cn1 + 128].copy()
    piece2 = grid[rn2:rn2 + 128, cn2:cn2 + 128].copy()

    grid[rn1:rn1 + 128, cn1:cn1 + 128] = piece2
    grid[rn2:rn2 + 128, cn2:cn2 + 128] = piece1

    return grid

def simulated_annealing(grid, initial_temp, cooling_rate, min_temp):
    current_grid = grid.copy()
    current_energy = calculate_energy(current_grid)
    best_grid = current_grid.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    
    while temperature > min_temp:
        new_grid = current_grid.copy()
        swap_pieces(new_grid)

        new_energy = calculate_energy(new_grid)
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
    return best_grid