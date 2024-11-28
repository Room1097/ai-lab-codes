import numpy as np

class GbikeRental:
    def __init__(self, max_bikes=20, max_move=5, rental_rate_1=3, rental_rate_2=4,
                 return_rate_1=3, return_rate_2=2, move_cost=2, rent_reward=10, discount=0.9,
                 free_move=0, parking_cost=4, parking_limit=10):
        self.max_bikes = max_bikes
        self.max_move = max_move
        self.rental_rate_1 = rental_rate_1
        self.rental_rate_2 = rental_rate_2
        self.return_rate_1 = return_rate_1
        self.return_rate_2 = return_rate_2
        self.move_cost = move_cost
        self.rent_reward = rent_reward
        self.discount = discount
        self.free_move = free_move
        self.parking_cost = parking_cost
        self.parking_limit = parking_limit

        self.states = [(i, j) for i in range(max_bikes + 1) for j in range(max_bikes + 1)]
        self.actions = range(-max_move, max_move + 1)  # Actions: bikes moved between locations
        self.value_function = np.zeros((max_bikes + 1, max_bikes + 1))  # State values
        self.policy = np.zeros((max_bikes + 1, max_bikes + 1), dtype=int)  # Initial policy

    def poisson(self, n, lam):
        """Compute Poisson probability."""
        return (np.exp(-lam) * lam ** n) / np.math.factorial(n)

    def expected_rentals(self, bikes, rental_rate):
        """Calculate expected rentals based on current bikes and rental rate."""
        expected_rentals = 0
        for rented in range(bikes + 1):
            prob = self.poisson(rented, rental_rate)
            expected_rentals += rented * prob
        return expected_rentals

    def state_transition(self, state, action):
        """Determine the state after an action and the corresponding cost."""
        # Bikes moved between locations
        bikes_loc1 = min(max(state[0] - action, 0), self.max_bikes)
        bikes_loc2 = min(max(state[1] + action, 0), self.max_bikes)

        # Apply free bike movement (one free move to location 2)
        move_cost = max(0, abs(action) - self.free_move) * self.move_cost

        # Apply parking constraints
        parking_cost = 0
        if bikes_loc1 > self.parking_limit:
            parking_cost += self.parking_cost
        if bikes_loc2 > self.parking_limit:
            parking_cost += self.parking_cost

        return (bikes_loc1, bikes_loc2), -(move_cost + parking_cost)

    def policy_evaluation(self, theta=1e-3):
        """Evaluate the value function under the current policy."""
        while True:
            delta = 0
            new_value_function = self.value_function.copy()

            for state in self.states:
                action = self.policy[state[0], state[1]]
                next_state, move_parking_cost = self.state_transition(state, action)

                # Compute expected reward
                reward = move_parking_cost
                reward += self.rent_reward * self.expected_rentals(state[0], self.rental_rate_1)
                reward += self.rent_reward * self.expected_rentals(state[1], self.rental_rate_2)

                # Bellman update
                new_value = reward + self.discount * self.value_function[next_state[0], next_state[1]]
                new_value_function[state[0], state[1]] = new_value

                delta = max(delta, abs(new_value - self.value_function[state[0], state[1]]))

            self.value_function = new_value_function
            if delta < theta:
                break

    def policy_improvement(self):
        """Improve the policy based on the current value function."""
        policy_stable = True
        for state in self.states:
            old_action = self.policy[state[0], state[1]]

            # Evaluate all possible actions
            action_values = {}
            for action in self.actions:
                if 0 <= state[0] - action <= self.max_bikes and 0 <= state[1] + action <= self.max_bikes:
                    next_state, move_parking_cost = self.state_transition(state, action)

                    reward = move_parking_cost
                    reward += self.rent_reward * self.expected_rentals(state[0], self.rental_rate_1)
                    reward += self.rent_reward * self.expected_rentals(state[1], self.rental_rate_2)

                    action_values[action] = reward + self.discount * self.value_function[next_state[0], next_state[1]]

            # Choose the best action
            best_action = max(action_values, key=action_values.get)
            self.policy[state[0], state[1]] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy_stable

    def policy_iteration(self):
        """Run policy iteration to find the optimal policy."""
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.value_function

# Example Usage
if __name__ == "__main__":
    # Initialize the GbikeRental problem
    gbike = GbikeRental(free_move=1, parking_cost=4, parking_limit=10)

    # Run policy iteration
    optimal_policy, optimal_value_function = gbike.policy_iteration()

    # Display results
    print("Optimal Policy:")
    print(optimal_policy)
    print("\nOptimal Value Function:")
    print(optimal_value_function)
