function value = compute_value(s1, s2, action, V, max_bikes, max_move, rental_rate1, ...
                                rental_rate2, return_rate1, return_rate2, move_cost, ...
                                rent_reward, gamma, free_move, parking_cost, parking_limit)
    % Transition due to action
    s1_after = max(0, min(max_bikes, s1 - action));
    s2_after = max(0, min(max_bikes, s2 + action));

    % Compute costs
    move_cost_actual = max(0, abs(action) - free_move) * move_cost;
    parking_cost_actual = 0;
    if s1_after > parking_limit
        parking_cost_actual = parking_cost_actual + parking_cost;
    end
    if s2_after > parking_limit
        parking_cost_actual = parking_cost_actual + parking_cost;
    end
    cost = -move_cost_actual - parking_cost_actual;

    % Expected rewards
    reward1 = compute_expected_reward(s1, rental_rate1, rent_reward);
    reward2 = compute_expected_reward(s2, rental_rate2, rent_reward);

    % Bellman equation
    value = cost + reward1 + reward2 + gamma * V(s1_after + 1, s2_after + 1);
end
