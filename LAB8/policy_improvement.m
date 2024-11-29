function [policy, stable] = policy_improvement(policy, V, max_bikes, max_move, ...
                                               rental_rate1, rental_rate2, return_rate1, ...
                                               return_rate2, move_cost, rent_reward, ...
                                               gamma, free_move, parking_cost, parking_limit)
    stable = true;
    for s1 = 0:max_bikes
        for s2 = 0:max_bikes
            old_action = policy(s1 + 1, s2 + 1);
            action_values = zeros(1, 2 * max_move + 1);
            actions = -max_move:max_move;

            for i = 1:length(actions)
                action = actions(i);
                if 0 <= s1 - action && s1 - action <= max_bikes && ...
                   0 <= s2 + action && s2 + action <= max_bikes
                    action_values(i) = compute_value(s1, s2, action, V, max_bikes, ...
                                                     max_move, rental_rate1, rental_rate2, ...
                                                     return_rate1, return_rate2, move_cost, ...
                                                     rent_reward, gamma, free_move, ...
                                                     parking_cost, parking_limit);
                else
                    action_values(i) = -inf; % Invalid action
                end
            end

            [~, best_action_idx] = max(action_values);
            policy(s1 + 1, s2 + 1) = actions(best_action_idx);

            if old_action ~= policy(s1 + 1, s2 + 1)
                stable = false;
            end
        end
    end
end
