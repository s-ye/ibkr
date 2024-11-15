# We are going to model a given stock price using Geometric Brownian Motion. 
# We will learn on historical data to determine a distribution of possible future prices.
# Our strategy will then be to buy when the price is below the expected price - (std_dev * threshold)
# and sell when the price is above the expected price + (std_dev * threshold).

# There are two ways that I understand how to do this:
# One is with Bayesian inference and Markov Chain Monte Carlo (MCMC) methods.
# The other is with Bayesian Neural Networks (BNNs).
# We are going to model a given stock price using Geometric Brownian Motion. 
# We will learn on historical data to determine a distribution of possible future prices.
# Our strategy will then be to buy when the price is below the expected price - (std_dev * threshold)
# and sell when the price is above the expected price + (std_dev * threshold).

import pymc as pm
import numpy as np 
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import itertools

class GBMModel:
    def __init__(self, historical_data):
        self.data = historical_data
        self.model = None
        self.trace = None
        self.mu_mean = .01
        self.mu_std = .1
        self.sigma_scale = .1
        

    def fit(self):
        returns = np.log(self.data['close']).diff().dropna()
        
        with pm.Model() as self.model:
            # Priors for mu and sigma
            mu = pm.Normal('mu', mu=self.mu_mean, sigma=self.mu_std)
            sigma = pm.HalfNormal('sigma', sigma=self.sigma_scale)
            
            # Likelihood of observed returns
            likelihood = pm.Normal('returns', mu=mu, sigma=sigma, observed=returns)
            
            # MCMC sampling
            self.trace = pm.sample(
                1000,               # reduce total samples
                tune=500,           # reduce tuning steps
                target_accept=0.8,  # lower target acceptance
                chains=4,           # fewer chains
                cores=4             # parallelize on 4 cores
            )
    def simulate_future_prices(self, start_price, time_periods, num_simulations=100):
        simulations = np.zeros((num_simulations, time_periods))
        
        for i in range(num_simulations):
            mu_sample = np.random.choice(self.trace.posterior['mu'].values.flatten())
            sigma_sample = np.random.choice(self.trace.posterior['sigma'].values.flatten())
            prices = [start_price]
            
            for t in range(1, time_periods):
                # 26 bars in a day
                dt = 1/26 

                next_price = prices[-1] * np.exp((mu_sample - 0.5 * sigma_sample**2) * dt +
                                                 sigma_sample * np.sqrt(dt) * np.random.normal())
                prices.append(next_price)
            
            simulations[i, :] = prices
        
        return simulations
    
    def format_func(value, tick_number):
        """Format function to make numbers more readable."""
        return f'{value:.4f}'  # Adjust decimal places as needed

    # Assuming you have a function to plot the traces:
    def plot_trace(self):
        trace = self.trace
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, var in enumerate(['mu', 'sigma']):
            # Density plot
            pm.plot_posterior(trace, var_names=[var], ax=axes[i * 2])
            
            # Trace plot
            pm.plot_trace(trace, var_names=[var])
            
            # Apply formatter
            for ax in axes[i * 2:i * 2 + 2]:
                ax.xaxis.set_major_formatter(FuncFormatter(self.format_func))
        
        plt.tight_layout()
        plt.show()

class GBMGridSearch:
    def __init__(self, historical_data, mu_means, mu_stds, sigma_scales, num_samples,burn_in_periods,chains):
        self.data = historical_data
        self.mu_means = mu_means
        self.mu_stds = mu_stds
        self.sigma_scales = sigma_scales
        self.num_samples = num_samples
        self.burn_in_periods = burn_in_periods
        self.chains = chains
        self.results = []

        self.grid = list(itertools.product(mu_means, mu_stds, sigma_scales, num_samples, burn_in_periods, chains))

    def fit(self,mu_mean,mu_std, sigma_scale, num_sample,burn_in,chain):
        results = np.log(self.data['close']).diff().dropna()

        with pm.Model() as model:
            # Priors for mu and sigma
            mu = pm.Normal('mu', mu=mu_mean, sigma=mu_std)
            sigma = pm.HalfNormal('sigma', sigma=sigma_scale)

            # Likelihood of observed returns
            likelihood = pm.Normal('returns', mu=mu, sigma=sigma, observed=results)

            # MCMC sampling
            trace = pm.sample(num_sample, tune=burn_in, target_accept=0.9, chains=chain)
            return trace

    def calculate_predictive_accuracy(self, trace, start_price, actual_prices, time_periods, num_simulations=150):
        """
        Calculate predictive accuracy metrics for the model.
        
        Parameters:
        - trace: The MCMC trace from sampling.
        - start_price: The initial price from which to simulate forward.
        - actual_prices: Array of actual observed prices for the forecast period.
        - time_periods: Number of future periods to simulate.
        - num_simulations: Number of simulations to run per future period.
        
        Returns:
        - A dictionary containing MAE, RMSE, and coverage probability.
        """
        # Step 1: Simulate future prices using the trace
        simulations = np.zeros((num_simulations, time_periods))
        
        for i in range(num_simulations):
            mu_sample = np.random.choice(trace.posterior['mu'].values.flatten())
            sigma_sample = np.random.choice(trace.posterior['sigma'].values.flatten())
            prices = [start_price]
            
            for t in range(1, time_periods):
                dt = 1/26
                next_price = prices[-1] * np.exp((mu_sample - 0.5 * sigma_sample**2) * dt +
                                                 sigma_sample * np.sqrt(dt) * np.random.normal())
                prices.append(next_price)
            
            simulations[i, :] = prices
        
        # Calculate the mean forecasted price for each period
        forecasted_prices = np.mean(simulations, axis=0)
        
        # Step 2: Calculate MAE and RMSE
        mae = np.mean(np.abs(forecasted_prices - actual_prices[:time_periods]))
        rmse = np.sqrt(np.mean((forecasted_prices - actual_prices[:time_periods]) ** 2))
        
        # Step 3: Calculate coverage probability within 95% prediction interval
        lower_bound = np.percentile(simulations, 2.5, axis=0)
        upper_bound = np.percentile(simulations, 97.5, axis=0)
        coverage = np.mean((actual_prices[:time_periods] >= lower_bound) & (actual_prices[:time_periods] <= upper_bound))
        
        return {'MAE': mae, 'RMSE': rmse, 'Coverage Probability': coverage}
    def run_grid_search(self,test_data):
        # test data is the data that we will use to test the model
        for mu_mean, mu_std, sigma_scale, num_sample, burn_in, chain in self.grid:
            trace = self.fit(mu_mean, mu_std, sigma_scale, num_sample, burn_in, chain)
            rhat_values = pm.rhat(trace).to_array().values
            if np.all(rhat_values <= 1.05):  # Adjust threshold if necessary
                # proceed to calculate accuracy and append results
                results = self.calculate_predictive_accuracy(trace, self.data['close'].iloc[-1], test_data['close'], len(test_data))
                self.results.append({
                    'mu_mean': mu_mean,
                    'mu_std': mu_std,
                    'sigma_scale': sigma_scale,
                    'num_sample': num_sample,
                    'burn_in': burn_in,
                    'chain': chain,
                    'MAE': results['MAE'],
                    'RMSE': results['RMSE'],
                    'Coverage Probability': results['Coverage Probability']
                })

    def log_results(self):
        results_df = pd.DataFrame(self.results)
        results_df.to_csv("results/grid_search_results.csv")
        print("Grid search results saved to 'results/grid_search_results.csv'")

        # save the 3 results with the lowest MAE and RMSE
        best_results_1 = results_df.nsmallest(3, ['MAE'])
        best_results_1.to_csv("results/best_results.csv")
        best_results_2 = results_df.nsmallest(3, ['RMSE'])
        best_results_2.to_csv("results/best_results.csv", mode='a', header=False)

        print("Best results saved to 'results/best_results.csv'")


