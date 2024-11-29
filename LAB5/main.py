import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

class RobustGaussianHMM:
    def __init__(self, n_states, max_iter=200, tol=1e-6, random_state=42):
        """
        Robust Gaussian Hidden Markov Model

        Parameters:
        - n_states: Number of hidden states
        - max_iter: Maximum EM iterations
        - tol: Convergence tolerance
        - random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters
        self.initial_prob = None
        self.transition_matrix = None
        self.means = None
        self.covars = None

    def _initialize_parameters(self, X):
        """
        Robust parameter initialization using clustering
        """
        np.random.seed(self.random_state)

        # Use k-means style initialization
        # Randomly select initial centroids
        indices = np.random.choice(len(X), self.n_states, replace=False)
        self.means = X[indices]

        # Initial covariances based on data spread
        self.covars = np.ones(self.n_states) * np.var(X)

        # Uniform initial probabilities and transition matrix
        self.initial_prob = np.ones(self.n_states) / self.n_states
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states

    def _gaussian_log_pdf(self, x, mu, sigma):
        """
        Compute log Gaussian probability density
        Prevents numerical instability
        """
        # Ensure positive variance
        sigma = max(sigma, 1e-6)

        # Log-space Gaussian PDF
        return norm.logpdf(x, loc=mu, scale=np.sqrt(sigma))

    def fit(self, X):
        """
        Expectation-Maximization (EM) algorithm for HMM
        """
        # Ensure X is 1D array
        X = X.flatten()

        # Initialize parameters
        self._initialize_parameters(X)

        # EM Algorithm
        for iter in range(self.max_iter):
            # Expectation Step
            # Compute log probabilities of observations
            log_likelihood_matrix = np.zeros((len(X), self.n_states))
            for j in range(self.n_states):
                log_likelihood_matrix[:, j] = [
                    self._gaussian_log_pdf(x, self.means[j], self.covars[j])
                    for x in X
                ]

            # Compute posterior probabilities
            log_initial_prob = np.log(self.initial_prob + 1e-300)
            log_transition_matrix = np.log(self.transition_matrix + 1e-300)

            # Forward-Backward algorithm requires more complex implementation
            # This is a simplified approximation
            posterior_probs = np.exp(log_likelihood_matrix -
                                     log_likelihood_matrix.max(axis=1)[:, np.newaxis])
            posterior_probs /= posterior_probs.sum(axis=1)[:, np.newaxis]

            # Maximization Step
            # Update means
            for j in range(self.n_states):
                self.means[j] = np.average(X, weights=posterior_probs[:, j])

            # Update covariances
            for j in range(self.n_states):
                self.covars[j] = np.average((X - self.means[j])**2,
                                            weights=posterior_probs[:, j])

            # Update initial probabilities
            self.initial_prob = posterior_probs[0]

            # Update transition matrix (simplified)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = np.mean(
                        posterior_probs[:-1, i] * posterior_probs[1:, j]
                    )

            # Normalize transition matrix
            self.transition_matrix /= self.transition_matrix.sum(axis=1)[:, np.newaxis]

        return self

    def _viterbi(self, X):
        """
        Viterbi algorithm for decoding the most likely sequence of hidden states.
        """
        # Ensure X is 1D array
        X = X.flatten()

        # Compute log probabilities of observations
        log_likelihood_matrix = np.zeros((len(X), self.n_states))
        for j in range(self.n_states):
            log_likelihood_matrix[:, j] = [
                self._gaussian_log_pdf(x, self.means[j], self.covars[j])
                for x in X
            ]

        # Initialization step
        log_delta = np.zeros((len(X), self.n_states))
        log_delta[0] = log_likelihood_matrix[0] + np.log(self.initial_prob)

        # Viterbi recursion step
        for t in range(1, len(X)):
            for j in range(self.n_states):
                log_delta[t, j] = log_likelihood_matrix[t, j] + np.max(
                    log_delta[t - 1] + np.log(self.transition_matrix[:, j])
                )

        # Backtracking to find the most likely sequence of states
        hidden_states = np.zeros(len(X), dtype=int)
        hidden_states[-1] = np.argmax(log_delta[-1])

        for t in range(len(X) - 2, -1, -1):
            hidden_states[t] = np.argmax(
                log_delta[t] + np.log(self.transition_matrix[:, hidden_states[t + 1]])
            )

        return hidden_states

    def predict(self, X):
        """
        Predict most likely hidden states using the Viterbi algorithm.
        """
        return self._viterbi(X)

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']]
    data['Daily Returns'] = data['Adj Close'].pct_change()
    data = data.dropna()
    return data

# Visualization function
def plot_hidden_states(data, hidden_states, ticker):
    plt.figure(figsize=(14, 8))
    for i in range(N_STATES):
        state_mask = hidden_states == i
        plt.plot(data.index[state_mask], data['Adj Close'][state_mask], '.', label=f"Hidden State {i}")
    plt.title(f"Hidden Market Regimes for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

# Parameters for the analysis
TICKER = "AAPL"  # Example: Apple Inc.
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Download data
data = download_stock_data(TICKER, START_DATE, END_DATE)

# Fit custom Gaussian HMM
N_STATES = 2
hmm_model = RobustGaussianHMM(n_states=N_STATES)

# Compute daily returns
returns = data['Daily Returns'].values

# Fit the model
hmm_model.fit(returns)

# Predict hidden states
hidden_states = hmm_model.predict(returns)

# Add hidden states to the dataset
data['Hidden State'] = hidden_states

# Plotting
plot_hidden_states(data, hidden_states, TICKER)

# Print model details
print("\nHidden State Details:")
for i in range(N_STATES):
    print(f"State {i}: Mean = {hmm_model.means[i]:.6f}, Variance = {hmm_model.covars[i]:.6f}")

# Transition Matrix
print("\nTransition Matrix:")
print(hmm_model.transition_matrix)

# State Distribution
print("\nState Distribution:")
unique, counts = np.unique(hidden_states, return_counts=True)
for state, count in zip(unique, counts):
    print(f"State {state}: {count} periods ({count/len(hidden_states)*100:.2f}%)")