% Gbike Bicycle Rental Problem with Modifications
% Policy Iteration Approach

clear;
close all;

Lamda = [3 4]; % Rental request rates at Location 1 and 2
lamda = [3 2]; % Return rates at Location 1 and 2
r = 10;        % Rental reward (INR)
t = 2;         % Transfer cost (INR)
gam = 0.9;     % Discount factor
theta = 0.1;   % Convergence threshold
free_transfer = 1; % Free bike transfer from Location 1 to 2
parking_limit = 10; % Maximum bikes without parking cost
parking_cost = 4;   % Parking cost if limit exceeded

% Initialize policy and value function
policy = zeros(21, 21); % No transfer as initial policy
policy_stable = false;
count = 0;


while ~policy_stable
    % Policy Evaluation
    V = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam, theta, free_transfer, parking_limit, parking_cost);

    % Policy Improvement
    [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, free_transfer, parking_limit, parking_cost);

    count = count + 1;
    fprintf('Iteration %d complete.\n', count);
end

% Visualization
figure;
subplot(2, 1, 1);
contour(policy, -5:5);
colorbar;
title('Optimal Policy (Bikes Transferred)');
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');

subplot(2, 1, 2);
surf(V);
colorbar;
title('Optimal Value Function');
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');
zlabel('Value');

disp(V);
disp(policy);