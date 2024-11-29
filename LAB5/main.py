import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

class SimpleHMM:
    def __init__(self, s, m=100, t=1e-5, r=42):
        self.s = s
        self.m = m
        self.t = t
        self.r = r
        self.pi = None
        self.tm = None
        self.mu = None
        self.var = None

    def init_params(self, data):
        np.random.seed(self.r)
        idx = np.random.choice(len(data), self.s, replace=False)
        self.mu = data[idx]
        self.var = np.ones(self.s) * np.var(data)
        self.pi = np.ones(self.s) / self.s
        self.tm = np.ones((self.s, self.s)) / self.s

    def g_log(self, x, m, v):
        v = max(v, 1e-6)
        return norm.logpdf(x, loc=m, scale=np.sqrt(v))

    def fit(self, d):
        d = d.flatten()
        self.init_params(d)
        for _ in range(self.m):
            llik = np.zeros((len(d), self.s))
            for j in range(self.s):
                llik[:, j] = [self.g_log(x, self.mu[j], self.var[j]) for x in d]
            pp = np.exp(llik - llik.max(axis=1)[:, None])
            pp /= pp.sum(axis=1)[:, None]
            for j in range(self.s):
                self.mu[j] = np.average(d, weights=pp[:, j])
                self.var[j] = np.average((d - self.mu[j])**2, weights=pp[:, j])
            self.pi = pp[0]
            for i in range(self.s):
                for j in range(self.s):
                    self.tm[i, j] = np.mean(pp[:-1, i] * pp[1:, j])
            self.tm /= self.tm.sum(axis=1)[:, None]

    def viterbi(self, d):
        d = d.flatten()
        llik = np.zeros((len(d), self.s))
        for j in range(self.s):
            llik[:, j] = [self.g_log(x, self.mu[j], self.var[j]) for x in d]
        dp = np.zeros((len(d), self.s))
        dp[0] = llik[0] + np.log(self.pi)
        for t in range(1, len(d)):
            for j in range(self.s):
                dp[t, j] = llik[t, j] + np.max(dp[t - 1] + np.log(self.tm[:, j]))
        hs = np.zeros(len(d), dtype=int)
        hs[-1] = np.argmax(dp[-1])
        for t in range(len(d) - 2, -1, -1):
            hs[t] = np.argmax(dp[t] + np.log(self.tm[:, hs[t + 1]]))
        return hs


tkr = "GOOGL"
sd = "2010-01-01"
ed = datetime.today().strftime('%Y-%m-%d')
st_data = yf.download(tkr, start=sd, end=ed)
st_data = st_data[['Adj Close']].dropna()
p = st_data['Adj Close'].values

hmm = SimpleHMM(s=2)
hmm.fit(p)
hs = hmm.viterbi(p)
st_data['state'] = hs

plt.figure(figsize=(14, 8))
for i in range(2):
    mask = hs == i
    plt.plot(st_data.index[mask], st_data['Adj Close'][mask], '.', label=f"State {i}")
plt.legend()
plt.show()

print("\nState Info:")
for i in range(2):
    print(f"State {i}: Mean = {hmm.mu[i]:.3f}, Var = {hmm.var[i]:.3f}")

print("\nTM:")
print(hmm.tm)

print("\nState Counts:")
for i, c in zip(*np.unique(hs, return_counts=True)):
    print(f"State {i}: {c} periods")
