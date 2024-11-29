function [V] = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam, theta, ...
                                       free_transfer, parking_limit, parking_cost)
    [m, n] = size(policy);
    V = zeros(m, n);

    % Precompute Poisson probabilities
    nn = 0:n-1;
    P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
    P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
    P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
    P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

    delta = theta + 1;
    while delta > theta
        v = V;
        delta = 0;

        for i = 1:m
            for j = 1:n
                s1 = i - 1; % State at Location 1
                s2 = j - 1; % State at Location 2
                a = policy(i, j); % Action (transfer bikes)
                R = -max(0, abs(a) - free_transfer) * t; % Adjusted transfer cost
                Vs_ = 0;

                % Transition after action
                s1_ = s1 - a;
                s2_ = s2 + a;

                % Parking costs
                if s1_ > parking_limit
                    R = R - parking_cost;
                end
                if s2_ > parking_limit
                    R = R - parking_cost;
                end

                for n1 = 0:12
                    for n2 = 0:14
                        s1__ = s1_ - min(n1, s1_);
                        s2__ = s2_ - min(n2, s2_);
                        for n3 = 0:12
                            for n4 = 0:9
                                s1___ = s1__ + min(n3, 20 - s1__);
                                s2___ = s2__ + min(n4, 20 - s2__);
                                Vs_ = Vs_ + P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * ...
                                      v(s1___ + 1, s2___ + 1);
                                R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * ...
                                         (min(n1, s1_) + min(n2, s2_))) * r;
                            end
                        end
                    end
                end

                V(i, j) = R + gam * Vs_;
                delta = max(delta, abs(v(i, j) - V(i, j)));
            end
        end
    end
end
