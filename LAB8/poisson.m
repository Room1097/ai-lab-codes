function prob = poisson(k, lambda)
    prob = exp(-lambda) * lambda^k / factorial(k);
end
