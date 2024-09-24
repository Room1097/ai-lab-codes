from functions.helpers import load_file
from functions.helpers import simulated_annealing
from PIL import Image 
import numpy as np

matrix = load_file("lena.txt")
N = 512

temperature_initial = 1000
cooling_rate = 0.99
threshold = 2
min_temp = 0.01

reconstructed_image = simulated_annealing(matrix, temperature_initial, cooling_rate, threshold, min_temp)

img = Image.fromarray(reconstructed_image.astype(np.uint8), mode='L')
img.save("reconstructed_image.png")