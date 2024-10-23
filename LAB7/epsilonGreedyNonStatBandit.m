function epsilonGreedyNonStatBandit()
    % Parameters
    epsilon = 0.1;  % Exploration rate
    alpha = 0.1;    % Step-size parameter for non-stationary rewards
    numActions = 10;  % Ten possible actions
    numIterations = 10000;  % Total steps
    totalReward = 0;  % Total reward

    % Initialize action-value estimates
    Q = zeros(1, numActions);  % Estimated values for each action

    % Initialize action selection counts (for visualization or debug purposes)
    actionCount = zeros(1, numActions);

    % Store average reward over time
    avgReward = zeros(1, numIterations);  % To track the average reward at each step

    % Run the simulation for a set number of iterations
    for t = 1:numIterations
        % --- Select action using epsilon-greedy strategy ---
        if rand < epsilon
            % Exploration: Choose a random action
            action = randi(numActions);
        else
            % Exploitation: Choose the best-known action
            [~, action] = max(Q);
        end

        % Get reward from the non-stationary bandit for the chosen action
        reward = bandit_nonstat(action);

        % Update total reward
        totalReward = totalReward + reward;

        % Increment the count for the chosen action (for debug/visualization)
        actionCount(action) = actionCount(action) + 1;

        % --- Update action-value estimate using a constant step-size ---
        Q(action) = Q(action) + alpha * (reward - Q(action));

        % Store the average reward at this time step
        avgReward(t) = totalReward / t;  % Average reward so far
    end
    
    % Display final estimated action values, action counts, and total reward
    disp('Final estimated action values:');
    disp(Q);
    disp('Action counts:');
    disp(actionCount);
    disp('Total reward after 10000 steps:');
    disp(totalReward);

    % --- Plot average reward over time ---
    figure;
    plot(avgReward, 'LineWidth', 2);
    xlabel('Time Steps');
    ylabel('Average Reward');
    title('Average Reward Over Time for Epsilon-Greedy Non-Stationary Bandit');
    grid on;
end
