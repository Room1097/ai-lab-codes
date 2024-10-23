function epsilonGreedyNonStatBandit1()
   % Parameters
   epsilon = 0.1;  % Exploration rate
   numActions = 10;  % Ten possible actions
   numIterations = 1000;  % Total steps
   totalReward = 0;  % Total reward

   % Initialize action-value estimates and counts
   Q = zeros(1, numActions);  % Estimated values
   actionCount = zeros(1, numActions);  % Count for each action

   % Store average reward over time
   avgReward = zeros(1, numIterations);  % To track average reward at each step

   for t = 1:numIterations
        % --- Select action ---
        if rand < epsilon
            % Exploration: Choose random action
            action = randi(numActions);
        else
            % Exploitation: Choose best-known action
            [~, action] = max(Q);
        end

        % Get reward from the bandit
        reward = bandit_nonstat(action);

        % Update total reward
        totalReward = totalReward + reward;

        % --- Update action-value estimate ---
        actionCount(action) = actionCount(action) + 1;
        Q(action) = Q(action) + (reward - Q(action)) / actionCount(action);

        % Store the average reward at this time step
        avgReward(t) = totalReward / t;  % Average reward so far
   end
    
   % Display final estimated action values and total reward
   disp('Final estimated action values:');
   disp(Q);
   disp('Total reward:');
   disp(totalReward);

   % --- Plot average reward over time ---
   figure;
   plot(avgReward, 'LineWidth', 2);
   xlabel('Time Steps');
   ylabel('Average Reward');
   title('Average Reward Over Time for Epsilon-Greedy Non-Stationary Bandit');
   grid on;
end
