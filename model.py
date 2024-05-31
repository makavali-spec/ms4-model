import numpy as np
import pandas as pd
from scipy.stats import norm, dirichlet, invgamma
import matplotlib.pyplot as plt

# Step 1: Load Data
data_path = 'C:/Users/Asus/Downloads/SP_SPX_1D.csv'
data = pd.read_csv(data_path)
data = data.dropna(axis =1 )
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
returns = data['close'].pct_change().dropna().values

# Define the MS-4 model
class MS4Model:
    def __init__(self, returns):
        self.returns = returns
        self.T = len(returns)
        
        # Initialize parameters
        self.mu = np.array([-0.007, 0.002, -0.002, 0.003])
        self.sigma2 = np.array([0.01, 0.01, 0.01, 0.01])
        self.P = np.array([[0.8, 0.1, 0.05, 0.05], 
                        [0.1, 0.8, 0.05, 0.05],
                        [0.05, 0.05, 0.8, 0.1], 
                        [0.05, 0.05, 0.1, 0.8]])
        
        # Priors
        self.mu_prior = [norm(-0.007, 0.1), norm(0.002, 0.1), norm(-0.002, 0.1), norm(0.003, 0.1)]
        self.sigma2_prior = [invgamma(1, scale=0.01), invgamma(1, scale=0.01), invgamma(1, scale=0.01), invgamma(1, scale=0.01)]
        self.P_prior = [dirichlet([8, 1.5, 0.5, 0.5]), dirichlet([1.5, 8, 0.5, 0.5]), 
                        dirichlet([0.5, 0.5, 8, 1.5]), dirichlet([0.5, 0.5, 1.5, 8])]
    def filter(self):
        # Hamilton filter to compute p(s_t|I_t)
        alpha = np.zeros((self.T, 4))
        alpha[0, :] = 1.0 / 4.0  # Equal initial probabilities

        for t in range(1, self.T):
            for j in range(4):
                prob = np.sum(alpha[t-1, :] * self.P[:, j])
                alpha[t, j] = prob * norm.pdf(self.returns[t], self.mu[j], np.sqrt(self.sigma2[j]))
            alpha[t, :] /= np.sum(alpha[t, :])
        
        self.alpha = alpha

    def smooth(self):
        # Forward-backward smoother to compute p(s_t|I_T)
        self.filter()
        beta = np.zeros((self.T, 4))
        beta[-1, :] = 1

        for t in range(self.T - 2, -1, -1):
            for i in range(4):
                beta[t, i] = np.sum(self.P[i, :] * norm.pdf(self.returns[t+1], self.mu, np.sqrt(self.sigma2)) * beta[t+1, :])
            beta[t, :] /= np.sum(beta[t, :])

        smoothed_probs = self.alpha * beta
        smoothed_probs /= smoothed_probs.sum(axis=1, keepdims=True)
        self.smoothed_probs = smoothed_probs
        return smoothed_probs

    def gibbs(self, niter):
        # Gibbs sampler to draw from the posterior
        np.random.seed(42)
        
        # Storage for the sampled parameters
        mu_samples = np.zeros((niter, 4))
        sigma2_samples = np.zeros((niter, 4))
        P_samples = np.zeros((niter, 4, 4))
        
        # Initial values
        s = np.zeros(self.T, dtype=int)
        
        for i in range(niter):
            # Draw s|M,Sigma,P using forward-backward algorithm
            self.smooth()
            for t in range(1, self.T):
                s[t] = np.argmax(np.random.multinomial(1, self.smoothed_probs[t, :]))
            
            # Draw M|Sigma,P,s
            for k in range(4):
                s_k = self.returns[s == k]
                n_k = len(s_k)
                mu_n = (np.sum(s_k) / self.sigma2[k] + self.mu_prior[k].mean() / self.mu_prior[k].var()) / (n_k / self.sigma2[k] + 1 / self.mu_prior[k].var())
                sigma_n = 1 / (n_k / self.sigma2[k] + 1 / self.mu_prior[k].var())
                self.mu[k] = np.random.normal(mu_n, np.sqrt(sigma_n))
                mu_samples[i, k] = self.mu[k]
            
            # Draw Sigma|M,P,s
            for k in range(4):
                s_k = self.returns[s == k]
                shape_n = 0.5 * (len(s_k) + 1)
                scale_n = 0.5 * (np.sum((s_k - self.mu[k])**2) + self.sigma2_prior[k].args[1])
                self.sigma2[k] = invgamma.rvs(shape_n, scale=scale_n)
                sigma2_samples[i, k] = self.sigma2[k]
            
            # Draw P|M,Sigma,s
            for k in range(4):
                counts = np.bincount(s[1:][s[:-1] == k], minlength=4)
                self.P[k, :] = np.random.dirichlet(self.P_prior[k].alpha + counts)
                P_samples[i, k, :] = self.P[k, :]
        
        # Compute posterior means
        self.mu_posterior = np.mean(mu_samples, axis=0)
        self.sigma2_posterior = np.mean(sigma2_samples, axis=0)
        self.P_posterior = np.mean(P_samples, axis=0)
        
        # Store smoothed state probabilities
        self.smoothed_probs = self.smooth()

    def predictive_density(self, h):
        # Compute the h-step ahead predictive density
        pred_density = np.zeros(h)
        state = np.argmax(self.smoothed_probs[-1])
        for i in range(h):
            next_state = np.argmax(np.random.multinomial(1, self.P[state, :]))
            pred_density[i] = np.random.normal(self.mu[next_state], np.sqrt(self.sigma2[next_state]))
            state = next_state
        return pred_density

    def value_at_risk(self, alpha, h=1):
        # Compute the Value-at-Risk for alpha level
        pred_density = self.predictive_density(h)
        var = np.percentile(pred_density, (1-alpha) * 100)
        return var

    def plot_regimes(self):
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(self.returns, label='Returns')
        for i in range(4):
            plt.fill_between(range(self.T), -0.1, 0.1, where=self.smoothed_probs[:, i] > 0.5, alpha=0.5, label=f'Regime {i+1}')
        plt.title('Returns and Regime Probabilities')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.show()

    def plot_cumulative_returns(self):
        cumulative_returns = np.cumsum(self.returns)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns')
        for i in range(4):
            plt.fill_between(range(self.T), cumulative_returns.min(), cumulative_returns.max(), where=self.smoothed_probs[:, i] > 0.5, alpha=0.3, label=f'Regime {i+1}')
        plt.title('Cumulative Returns with Regime Shifts')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()

# Example usage        
model = MS4Model(returns)
model.gibbs(100)
model.plot_regimes()
model.plot_cumulative_returns()

# Value at Risk example
var_95 = model.value_at_risk(0.95)
print(f"Value at Risk (95% confidence): {var_95}")
