function epsilonGreedyBandit()
    % Parameters
    epsilon = 0.1;  % Exploration rate
    numActions = 2;  % Two possible actions
    numIterations = 1000;  % Total steps
    totalRewardA = 0;  % Total reward for bandit A
    totalRewardB = 0;  % Total reward for bandit B
    
    % Optimal actions (manually set based on highest probability of success)
    optimalActionA = 2;  % Assume action 2 is optimal for Bandit A (based on reward probabilities)
    optimalActionB = 2;  % Assume action 2 is optimal for Bandit B

    % Track how often the optimal action is chosen
    optimalActionCountA = zeros(1, numIterations);
    optimalActionCountB = zeros(1, numIterations);

    % Initialize action-value estimates and counts for each bandit
    Q_A = zeros(1, numActions);  % Estimated values for bandit A
    Q_B = zeros(1, numActions);  % Estimated values for bandit B
    actionCount_A = zeros(1, numActions);  % Count for each action (bandit A)
    actionCount_B = zeros(1, numActions);  % Count for each action (bandit B)

    for t = 1:numIterations
        % --- Select action for bandit A ---
        if rand < epsilon
            % Exploration: Choose random action
            action_A = randi(numActions);
        else
            % Exploitation: Choose best-known action
            [~, action_A] = max(Q_A);
        end

        % --- Select action for bandit B ---
        if rand < epsilon
            % Exploration: Choose random action
            action_B = randi(numActions);
        else
            % Exploitation: Choose best-known action
            [~, action_B] = max(Q_B);
        end

        % Get rewards from both bandits
        reward_A = binaryBanditA(action_A);
        reward_B = binaryBanditB(action_B);

        % Update total rewards
        totalRewardA = totalRewardA + reward_A;
        totalRewardB = totalRewardB + reward_B;

        % --- Update action-value estimate for bandit A ---
        actionCount_A(action_A) = actionCount_A(action_A) + 1;
        Q_A(action_A) = Q_A(action_A) + (reward_A - Q_A(action_A)) / actionCount_A(action_A);

        % --- Update action-value estimate for bandit B ---
        actionCount_B(action_B) = actionCount_B(action_B) + 1;
        Q_B(action_B) = Q_B(action_B) + (reward_B - Q_B(action_B)) / actionCount_B(action_B);

        % --- Track optimal action selection ---
        optimalActionCountA(t) = action_A == optimalActionA;
        optimalActionCountB(t) = action_B == optimalActionB;
    end
    
    % Display final estimated action values and total rewards
    disp('Final estimated action values for Bandit A:');
    disp(Q_A);
    disp('Total reward for Bandit A:');
    disp(totalRewardA);
    
    disp('Final estimated action values for Bandit B:');
    disp(Q_B);
    disp('Total reward for Bandit B:');
    disp(totalRewardB);

    % --- Plot optimal action selection over time ---
    figure;
    hold on;
    plot(cumsum(optimalActionCountA) ./ (1:numIterations), 'b-', 'DisplayName', 'Bandit A');
    plot(cumsum(optimalActionCountB) ./ (1:numIterations), 'r-', 'DisplayName', 'Bandit B');
    xlabel('Time Steps');
    ylabel('Proportion of Optimal Action');
    title('Optimal Action Selection Over Time');
    legend show;
    grid on;
    hold off;
end
