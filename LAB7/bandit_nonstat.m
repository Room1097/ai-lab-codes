function [value] = bandit_nonstat(action)
    % Persistent variables to store the current means of the bandit's arms
    persistent means;
    persistent initialized;

    if isempty(initialized)
        % Initialize the means of the 10 arms to 0
        means = zeros(1, 10);
        initialized = true;
    end

    % Standard deviation for the random walk of the means
    std_dev_walk = 0.01;
    
    % Perform the random walk for each arm's mean
    means = means + normrnd(0, std_dev_walk, [1, 10]);
    
    % Choose the reward based on the current mean of the selected action
    % Assume the reward is drawn from a normal distribution with variance 1
    value = normrnd(means(action), 1);
end
