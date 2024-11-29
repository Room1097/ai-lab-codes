function reward = compute_expected_reward(bikes, rental_rate, rent_reward)
    reward = 0;
    for rented = 0:bikes
        prob = poisson(rented, rental_rate);
        reward = reward + rented * rent_reward * prob;
    end
end
