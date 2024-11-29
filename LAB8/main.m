% Gbike Bicycle Rental Problem

% Parameters
max_bikes = 20;        % Maximum bikes at each location
max_move = 5;          % Maximum bikes moved per night
rental_rate1 = 3;      % Rental request rate at location 1
rental_rate2 = 4;      % Rental request rate at location 2
return_rate1 = 3;      % Return rate at location 1
return_rate2 = 2;      % Return rate at location 2
move_cost = 2;         % Cost per bike moved
rent_reward = 10;      % Reward per bike rented
gamma = 0.9;           % Discount factor
theta = 1e-3;          % Convergence threshold

% Modifications
free_move = 1;         % Number of free bikes moved
parking_cost = 4;      % Parking cost
parking_limit = 10;    % Maximum bikes without parking cost

% Initialize policy and value function
policy = zeros(max_bikes + 1, max_bikes + 1); % Policy: actions
value_function = zeros(max_bikes + 1, max_bikes + 1); % Value: V(s)

% Policy Iteration
while true
    % Policy Evaluation
    value_function = policy_evaluation(policy, value_function, max_bikes, max_move, ...
                                       rental_rate1, rental_rate2, return_rate1, ...
                                       return_rate2, move_cost, rent_reward, gamma, ...
                                       free_move, parking_cost, parking_limit, theta);

    % Policy Improvement
    [policy, policy_stable] = policy_improvement(policy, value_function, max_bikes, ...
                                                 max_move, rental_rate1, rental_rate2, ...
                                                 return_rate1, return_rate2, move_cost, ...
                                                 rent_reward, gamma, free_move, ...
                                                 parking_cost, parking_limit);
    if policy_stable
        break;
    end
end

% Display Results
disp('Optimal Policy:');
disp(policy);
disp('Optimal Value Function:');
disp(value_function);

% Visualization
figure;

% 1. Value Function Heatmap
subplot(1, 2, 1);
imagesc(0:max_bikes, 0:max_bikes, value_function);
colorbar;
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');
title('Value Function Heatmap');
axis square;

% 2. Policy Quiver Plot
subplot(1, 2, 2);
[X, Y] = meshgrid(0:max_bikes, 0:max_bikes);
U = zeros(size(policy)); % Movement in X-direction (bikes to/from Location 2)
V = zeros(size(policy)); % Movement in Y-direction (bikes to/from Location 1)

for i = 1:size(policy, 1)
    for j = 1:size(policy, 2)
        action = policy(i, j);
        U(i, j) = action;     % Positive: move bikes to Location 2
        V(i, j) = -action;    % Negative: move bikes from Location 1
    end
end

quiver(X, Y, U, V, 'AutoScale', 'on', 'LineWidth', 1.5);
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');
title('Policy Quiver Plot');
axis([0 max_bikes 0 max_bikes]);
axis square;
