import numpy as np

class GridWorld:
    def __init__(self, rows, cols, terminal_states, rewards, transition_prob=0.8, gamma=1.0):
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states  
        self.rewards = rewards  
        self.transition_prob = transition_prob  
        self.gamma = gamma  
        self.actions = ['up', 'down', 'left', 'right']  
        self.value_function = np.zeros((rows, cols))  

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_state(self, state, action):
        row, col = state
        if action == 'up':
            return max(0, row - 1), col
        elif action == 'down':
            return min(self.rows - 1, row + 1), col
        elif action == 'left':
            return row, max(0, col - 1)
        elif action == 'right':
            return row, min(self.cols - 1, col + 1)

    def get_transitions(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state)]  

        transitions = []
        intended_state = self.get_next_state(state, action)
        transitions.append((self.transition_prob, intended_state))

        orthogonal_actions = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }
        for ortho_action in orthogonal_actions[action]:
            ortho_state = self.get_next_state(state, ortho_action)
            transitions.append(((1 - self.transition_prob) / 2, ortho_state))

        return transitions

    def value_iteration(self, threshold=1e-3):
        while True:
            delta = 0
            new_value_function = self.value_function.copy()

            for row in range(self.rows):
                for col in range(self.cols):
                    state = (row, col)

                    if self.is_terminal(state):
                        new_value_function[row, col] = self.terminal_states[state]
                        continue

                    action_values = []
                    for action in self.actions:
                        transitions = self.get_transitions(state, action)
                        action_value = sum(
                            prob * (self.rewards + self.gamma * self.value_function[next_state])
                            for prob, next_state in transitions
                        )
                        action_values.append(action_value)

                    new_value_function[row, col] = max(action_values)
                    delta = max(delta, abs(new_value_function[row, col] - self.value_function[row, col]))

            self.value_function = new_value_function
            if delta < threshold:
                break

        return self.value_function

    def print_policy(self):
        policy = np.empty((self.rows, self.cols), dtype=str)
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                if self.is_terminal(state):
                    policy[row, col] = 'T'
                    continue

                action_values = {}
                for action in self.actions:
                    transitions = self.get_transitions(state, action)
                    action_values[action] = sum(
                        prob * (self.rewards + self.gamma * self.value_function[next_state])
                        for prob, next_state in transitions
                    )
                policy[row, col] = max(action_values, key=action_values.get)

        print("\nOptimal Policy:")
        for row in policy:
            print(" ".join(row))


if __name__ == "__main__":
    rows, cols = 4, 3
    terminal_states = {(0, 2): 1, (1, 2): -1} 
    gamma = 0.9  

    reward_structures = [-2, 0.1, 0.02, 1]

    for reward in reward_structures:
        print(f"\n=== Value Iteration for Reward r(s) = {reward} ===")
        grid_world = GridWorld(rows, cols, terminal_states, reward, gamma=gamma)

        optimal_values = grid_world.value_iteration()
        print("Optimal Value Function:")
        print(optimal_values)

        grid_world.print_policy()
