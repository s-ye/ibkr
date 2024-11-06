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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class GBMModel:
    def __init__(self, historical_data):
        self.data = historical_data
        self.model = None
        self.trace = None
        # self.fit()
        # self.plot_trace()
        

    def fit(self):
        returns = np.log(self.data['close']).diff().dropna()
        
        with pm.Model() as self.model:
            # Priors for mu and sigma
            mu = pm.Normal('mu', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Likelihood of observed returns
            likelihood = pm.Normal('returns', mu=mu, sigma=sigma, observed=returns)
            
            # MCMC sampling
            self.trace = pm.sample(2000, tune=1000, target_accept=0.9)

    def simulate_future_prices(self, start_price, time_periods, num_simulations=100):
        simulations = np.zeros((num_simulations, time_periods))
        
        for i in range(num_simulations):
            mu_sample = np.random.choice(self.trace.posterior['mu'].values.flatten())
            sigma_sample = np.random.choice(self.trace.posterior['sigma'].values.flatten())
            prices = [start_price]
            
            for t in range(1, time_periods):
                dt = 1
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


# # Main code
# import pandas as pd
# from datetime import datetime
# from backtester import Backtester
# import matplotlib.pyplot as plt

# import matplotlib.dates as mdates

# if __name__ == '__main__':
#     # Load historical data
#     backtester = Backtester('AAPL', 'SMART', 'USD')
#     historical_data = backtester.full_data
#     gbm = GBMModel(historical_data)

#     # Simulate future prices
#     simulations = gbm.simulate_future_prices(start_price=historical_data['close'].values[-1], time_periods=30,num_simulations=10)

#     # Plot the last month of historical data
#     plt.figure(figsize=(12, 6))
#     plt.title('Historical Prices for AAPL')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()

#     plt.plot(historical_data.index[-90:], historical_data['close'].values[-90:], color='blue', label='Historical Data')

#     # Generate a timestamp range for future prices, starting from the last timestamp in historical data
#     future_times = pd.date_range(start=historical_data.index[-1], periods=30, freq='15T')
    
#     # Plot the simulations
#     for i in range(simulations.shape[0]):
#         plt.plot(future_times, simulations[i, :], color='red', alpha=0.5)

#     plt.show()
    
    
#     # Disconnect from IBKR
#     backtester.disconnect()

