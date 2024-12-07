# We screen the log returns of the data and evaluate how well it fits a normal distribution.
# If it is not suitable, we will need to adjust the priors for the model.
# Suitibility can be evaluated by looking at the posterior predictive checks.

from backtester import Backtester
import numpy as np
from scipy.stats import shapiro, kstest, norm
import matplotlib.pyplot as plt


stock = 'UAL'

if __name__ == "__main__":
    backtester = Backtester(stock, 'SMART', 'USD')
    # different time frames
    datas = [
        ('1 day 1 year', backtester.one_yr_1d_data),
        ('1 day 3 months', backtester.three_mo_1d_data),
        ('1mo 15 mins', backtester.one_mo_15min_data)
    ]
    for (name,data) in datas:
        log_returns = np.log(data['close']).diff().dropna()
        print(f"Screening log returns for {name} data")
        # Shapiro-Wilk test for normality
        stat, p_value = shapiro(log_returns)
        print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p_value}")

        # Kolmogorov-Smirnov test against normal distribution
        mean, std = log_returns.mean(), log_returns.std()
        stat, p_value = kstest(log_returns, 'norm', args=(mean, std))
        print(f"K-S Test Statistic: {stat}, p-value: {p_value}")

        # save histogram
        plt.hist(log_returns, bins=50, color='blue', alpha=0.5, density=True)
        plt.title(f"Log Returns Distribution: {name}")
        plt.xlabel("Log Returns")
        plt.ylabel("Density")
        plt.savefig(f"output/{stock}_log_returns_{name}.png")
        plt.clf()



