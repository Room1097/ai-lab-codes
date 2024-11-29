function [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, free_transfer, parking_limit, parking_cost)
    [m, n] = size(policy);

    nn = 0:n-1;
    P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
    P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
    P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
    P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

    policy_stable = true;

    for i = 1:m
        for j = 1:n
            s1 = i - 1; 
            s2 = j - 1; 
            amin = -min(min(s2, m - 1 - s1), 5);
            amax = min(min(s1, n - 1 - s2), 5);
            old_action = policy(i, j);
            best_value = -inf;

            for a = amin:amax
                R = -max(0, abs(a) - free_transfer) * t; 
                Vs_ = 0;

                s1_ = s1 - a;
                s2_ = s2 + a;

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
                                Vs_ = Vs_ + P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1);
                                R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                            end
                        end
                    end
                end

                value = R + gam * Vs_;
                if value > best_value
                    best_value = value;
                    policy(i, j) = a;
                end
            end

            if old_action ~= policy(i, j)
                policy_stable = false;
            end
        end
    end
end
