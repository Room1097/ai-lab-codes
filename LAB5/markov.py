import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

# Helper functions for Gaussian probability density
def gaussian_prob(x, mean, cov):
    return multivariate_normal(mean=mean, cov=cov).pdf(x)

# Step 1: Load Data
def download_stock_data(symbol, start, end):
    print("Downloading stock data...")
    stock_data = yf.download(symbol, start=start, end=end)
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data.dropna(subset=['Returns'])
    print("Stock data downloaded.")
    return stock_data

# Step 2: Initialize HMM Parameters
def initialize_hmm(n_states, n_features):
    print("Initializing HMM parameters...")
    np.random.seed(42)
    transition_matrix = np.full((n_states, n_states), 1.0 / n_states)
    means = np.random.rand(n_states, n_features)
    covariances = np.array([np.eye(n_features) for _ in range(n_states)])
    initial_probs = np.full(n_states, 1.0 / n_states)
    print("HMM parameters initialized.")
    return transition_matrix, means, covariances, initial_probs

# Step 3: Expectation-Maximization for HMM
def forward_backward_algorithm(X, transition_matrix, means, covariances, initial_probs):
    print("Running forward-backward algorithm...")
    n_states = len(initial_probs)
    n_samples = len(X)

    # Step 3a: Forward Pass
    log_alpha = np.zeros((n_samples, n_states))
    emission_probs = np.array([gaussian_prob(X[0], means[i], covariances[i]) for i in range(n_states)])
    log_alpha[0] = np.log(initial_probs) + np.log(emission_probs + 1e-16)  # Add small value to avoid log(0)

    for t in range(1, n_samples):
        emission_probs = np.array([gaussian_prob(X[t], means[j], covariances[j]) for j in range(n_states)])
        for j in range(n_states):
            log_alpha[t, j] = np.log(emission_probs[j] + 1e-16) + logsumexp(log_alpha[t-1] + np.log(transition_matrix[:, j] + 1e-16))

    # Step 3b: Backward Pass
    log_beta = np.zeros((n_samples, n_states))

    for t in reversed(range(n_samples - 1)):
        emission_probs = np.array([gaussian_prob(X[t + 1], means[j], covariances[j]) for j in range(n_states)])
        for i in range(n_states):
            log_beta[t, i] = logsumexp(np.log(transition_matrix[i] + 1e-16) + np.log(emission_probs + 1e-16) + log_beta[t + 1])

    # Step 3c: Posterior Probabilities (gamma)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

    print("Forward-backward algorithm completed.")
    return np.exp(log_gamma), log_alpha, log_beta

def em_algorithm(X, n_states, n_features, n_iter=10):
    print("Starting EM algorithm...")
    # Initialize parameters
    transition_matrix, means, covariances, initial_probs = initialize_hmm(n_states, n_features)

    for iteration in range(n_iter):
        print(f"EM iteration {iteration + 1}...")
        # E-step: Forward-Backward algorithm to compute posteriors
        gamma, log_alpha, log_beta = forward_backward_algorithm(X, transition_matrix, means, covariances, initial_probs)
        xi = np.zeros((len(X) - 1, n_states, n_states))

        for t in range(len(X) - 1):
            emission_probs = np.array([gaussian_prob(X[t + 1], means[j], covariances[j]) for j in range(n_states)])
            log_xi_t = np.log(transition_matrix + 1e-16) + np.log(emission_probs + 1e-16) + log_alpha[t].reshape(-1, 1) + log_beta[t + 1].reshape(1, -1)
            log_xi_denominator = logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t - log_xi_denominator)

        # M-step: Re-estimate the parameters using posteriors
        initial_probs = gamma[0]
        xi_sum = xi.sum(axis=0)  # shape (n_states, n_states)
        transition_matrix = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        means = np.dot(gamma.T, X) / gamma.sum(axis=0)[:, np.newaxis]

        for i in range(n_states):
            diff = X - means[i]
            covariances[i] = np.dot((gamma[:, i][:, np.newaxis] * diff).T, diff) / gamma[:, i].sum()

    print("EM algorithm completed.")
    return transition_matrix, means, covariances, initial_probs, gamma

# Step 4: Visualizing the Hidden States
def plot_hidden_states(stock_data, hidden_states):
    print("Plotting hidden states...")
    plt.figure(figsize=(14, 8))
    plt.plot(stock_data.index, stock_data['Returns'], label='Daily Returns')

    # Create an array to store hidden states aligned with stock_data
    hidden_states_full = np.full(len(stock_data), np.nan)

    # Align hidden_states with stock_data
    hidden_states_full[1:len(hidden_states) + 1] = hidden_states  # Shift hidden states to align with stock_data

    # Color-code based on hidden states
    for state in np.unique(hidden_states):
        state_mask = hidden_states_full[1:] == state  # Use hidden_states_full[1:] for masking
        plt.fill_between(stock_data.index[1:], stock_data['Returns'][1:], where=state_mask, alpha=0.3, label=f'State {state}')

    plt.legend()
    plt.title('Daily Returns with Hidden Market Regimes (Tesla)')
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.savefig('hidden_states_plot.png')
    print("Hidden states plot saved as 'hidden_states_plot.png'.")
    plt.show()

# Main function to run the HMM
if __name__ == "__main__":
    print("Starting main function...")
    # Download historical data for Tesla stock
    stock_data = download_stock_data('TSLA', start='2013-01-01', end='2023-01-01')
    returns = stock_data['Returns'].values.reshape(-1, 1)  # Daily returns as the observable variable

    # Fit HMM model with 2 hidden states
    n_states = 2
    n_features = 1
    transition_matrix, means, covariances, initial_probs, gamma = em_algorithm(returns, n_states, n_features, n_iter=100)

    # Decode hidden states
    hidden_states = np.argmax(gamma, axis=1)

    # Visualize the results
    plot_hidden_states(stock_data, hidden_states)
    print("Main function completed.")
