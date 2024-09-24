import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# Part 1: Data Collection and Preprocessing

# Download historical data for a chosen stock (e.g., Tesla)
symbol = "TSLA"
start_date = "2013-01-01"
end_date = "2023-01-01"

data = yf.download(symbol, start=start_date, end=end_date)

# Preprocess Data
data['Returns'] = data['Adj Close'].pct_change().dropna()
returns = data[['Returns']].dropna()

# Display the first few rows of the returns data
print("Returns Data:")
print(returns.head())

# Part 2: Gaussian Hidden Markov Model

# Define and fit the Gaussian HMM model
n_hidden_states = 2  # Adjust this for more states if needed
model = GaussianHMM(n_components=n_hidden_states, covariance_type="diag", n_iter=1000)
model.fit(returns)

# Predict hidden states
hidden_states = model.predict(returns)

# Part 3: Interpretation and Inference

# Analyze hidden states
means = model.means_

# Print the shape of covars_ for debugging
print(f"Shape of covars_: {model.covars_.shape}")

# Calculate variances based on the shape of covars_
if model.covars_.ndim == 1:
    variances = np.sqrt(model.covars_)
elif model.covars_.ndim == 2:
    variances = np.sqrt(np.diagonal(model.covars_, axis1=1, axis2=2))
elif model.covars_.ndim == 3:  # Handle the case for shape (n_states, 1, 1)
    variances = np.sqrt(model.covars_[:, 0, 0])
else:
    raise ValueError("Unexpected shape of covariance matrix")

# Ensure variances is defined before trying to use it
print("\nHidden States Analysis:")
for i in range(n_hidden_states):
    if 'variances' in locals():  # Check if variances exists
        print(f"State {i + 1}: Mean Return = {means[i][0]:.6f}, Variance = {variances[i]:.6f}")
    else:
        print("Variances not calculated.")

# Part 4: Visualization

# Create a scatter plot of returns colored by hidden states
plt.figure(figsize=(12, 6))
plt.plot(data.index[1:], returns, label="Returns", color='gray', alpha=0.5)  # Adjusted for length
plt.scatter(data.index[1:], returns, c=hidden_states, cmap='coolwarm', marker='o', s=10, label="Hidden States")
plt.title(f"Hidden Market Regimes for {symbol}")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.show()

# Transition Matrix
transition_matrix = model.transmat_
print("\nTransition Matrix:")
print(transition_matrix)

# Part 5: Conclusions and Insights

# Predict the next likely state based on the most recent hidden state
next_state = model.predict(returns[-1:])
print(f"\nThe next predicted hidden state: {next_state[0] + 1}")  # Adding 1 for human-readable format

# Optionally: Save the plot to a file
plt.savefig("hidden_market_regimes.png")

# Additional visualizations for insights (e.g., histogram of returns)
plt.figure(figsize=(12, 6))
plt.hist(returns['Returns'], bins=50, alpha=0.6, color='blue')
plt.title(f"Distribution of Returns for {symbol}")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.grid()
plt.show()
