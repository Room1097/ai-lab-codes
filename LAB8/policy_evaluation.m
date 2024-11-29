function V = policy_evaluation(policy, V, max_bikes, max_move, rental_rate1, rental_rate2, ...
                                return_rate1, return_rate2, move_cost, rent_reward, ...
                                gamma, free_move, parking_cost, parking_limit, theta)
    while true
        delta = 0;
        for s1 = 0:max_bikes
            for s2 = 0:max_bikes
                action = policy(s1 + 1, s2 + 1); % Current policy action
                new_value = compute_value(s1, s2, action, V, max_bikes, max_move, ...
                                          rental_rate1, rental_rate2, return_rate1, ...
                                          return_rate2, move_cost, rent_reward, ...
                                          gamma, free_move, parking_cost, parking_limit);
                delta = max(delta, abs(new_value - V(s1 + 1, s2 + 1)));
                V(s1 + 1, s2 + 1) = new_value;
            end
        end
        if delta < theta
            break;
        end
    end
end
