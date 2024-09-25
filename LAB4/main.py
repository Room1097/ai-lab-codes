import numpy as np
from PIL import Image

from functions.helpers import load_file, simulated_annealing, calculate_energy

N = 512
temperature_initial = 1000
cooling_rate = 0.90
min_temp = 0.01
num_iters = 10

grid = load_file("lena.txt")
img = Image.fromarray(grid.astype(np.uint8), 'L')
img.save('input.png')

minEnergy = float('inf')
res = []

for i in range(num_iters):
    solved_grid = simulated_annealing(grid, temperature_initial, cooling_rate, min_temp)
    energy = calculate_energy(solved_grid)

    if solved_grid is not None and energy < minEnergy:
        minEnergy = energy
        grid = solved_grid.copy()
        res = grid

    print(f"Iteration {i + 1}, Cost: {energy}")

    if solved_grid is not None:
        img = Image.fromarray(solved_grid.astype(np.uint8), 'L')
        img.save(f'output_iteration_{i + 1}.png')

img = Image.fromarray(grid.astype(np.uint8), 'L')
img.save('output_iteration_final.png')
