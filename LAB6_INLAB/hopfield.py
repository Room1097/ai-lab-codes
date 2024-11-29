import numpy as np
import matplotlib.pyplot as plt

N = 100
P = 5

patterns = np.random.choice([-1, 1], size=(P, N))

weights = np.zeros((N, N))
for p in patterns:
    weights += np.outer(p, p)
weights /= N
np.fill_diagonal(weights, 0)

def display_patterns(patterns, titles, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.ravel() if rows * cols > 1 else [axes]
    for i, (pattern, title) in enumerate(zip(patterns, titles)):
        axes[i].imshow(pattern.reshape(10, 10), cmap="binary")
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

def add_noise(pattern, noise_level=0.2):
    noisy_pattern = pattern.copy()
    flip_indices = np.random.choice([True, False], size=N, p=[noise_level, 1 - noise_level])
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def update_state(state, weights, iterations=10):
    for _ in range(iterations):
        for i in range(N):
            state[i] = np.sign(np.dot(weights[i], state))
    return state

test_pattern_index = 0
original_pattern = patterns[test_pattern_index]
noisy_pattern = add_noise(original_pattern, noise_level=0.3)

retrieved_pattern = update_state(noisy_pattern.copy(), weights)

all_patterns = [original_pattern for original_pattern in patterns]
all_patterns.append(noisy_pattern)
all_patterns.append(retrieved_pattern)

titles = [f"Stored Pattern {i + 1}" for i in range(P)]
titles.append("Noisy Pattern")
titles.append("Retrieved Pattern")

display_patterns(all_patterns, titles, rows=2, cols=(P // 2 + 2))
