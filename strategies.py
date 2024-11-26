# strategies.py
from base_strategy import BaseStrategy
import pandas as pd
import matplotlib.pyplot as plt
from stoploss_takeprofit_strategy import StopLossTakeProfitStrategy
from datetime import time


import numpy as np
import matplotlib.pyplot as plt
import gbm as gbm
import matplotlib.dates as mdates

class GBMStrategy(StopLossTakeProfitStrategy):
    def __init__(self, contract, data, ib, params,initial_capital=1_000_000, position_size_pct=0.02,profit_target_pct=0.05, trailing_stop_pct = 0.03):
        super().__init__(contract, data, ib, params, initial_capital=initial_capital, position_size_pct=position_size_pct, profit_target_pct=profit_target_pct, trailing_stop_pct=trailing_stop_pct)
        self.threshold = self.params.get('threshold', 1)
        self.time_periods = self.params.get('time_periods', 30)
        self.num_simulations = self.params.get('num_simulations', 100)
        self.position_size_pct = position_size_pct
        self.predictions = {}  # Dictionary to store predictions for plotting

    def generate_signals(self):
        """
        Generate buy/sell signals based on forecasted price distributions from GBM, fitting every 'self.time_periods' periods.
        """
        self.data['signal'] = 0  # Reset signals
        self.data['normalized_volume'] = self.data['volume'] / self.data['volume'].rolling(30).mean()  # Normalize volume
        num_periods = len(self.data)
        predictions = {}  # Dictionary to store predictions for each time_periods block
        min_data_points = self.time_periods  # Minimum data points required to fit the model
        frequency = self.data.index[-1] - self.data.index[-2]
        if frequency == pd.Timedelta('1 days'):
            frequency = 'B'
        else:
            frequency = '15T'


        def fit_model(start_idx):
            """
            Fit the model on data up to start_idx and generate 'self.time_periods' predictions.
            """
            restricted_data = self.data.iloc[:start_idx]
            
            # Check if restricted_data has enough data to proceed
            if len(restricted_data) < min_data_points:
                raise ValueError(f"Insufficient data to fit model at index {start_idx}")

            start_price = restricted_data['close'].iloc[-1]
            model = gbm.GBMModel(restricted_data)
            model.fit()
            # Generate predictions for the next 'self.time_periods' and cache them
            simulations = model.simulate_future_prices(start_price, frequency, self.time_periods, self.num_simulations)
            # Cache expected price and sigma for each of the 'self.time_periods' future periods
            return [(np.mean(simulations[:, j]), np.std(simulations[:, j])) for j in range(self.time_periods)]
        
        # Iterate over the data in chunks of 'self.time_periods'
        for start_idx in range(0, num_periods, self.time_periods):
            if start_idx < min_data_points:
                continue  # Skip to the next block if there's insufficient data

            end_idx = min(start_idx + self.time_periods, num_periods)
            # Fit the model and generate predictions for the next 'self.time_periods'
            predictions[start_idx] = fit_model(start_idx)

            # Generate signals based on the predictions
            for i in range(start_idx, end_idx):
                # Access the cached expected price and sigma for the period
                expected_price, sigma_price = predictions[start_idx][i - start_idx]
                buy_threshold = expected_price - (self.threshold * sigma_price)
                sell_threshold = expected_price + (self.threshold * sigma_price)
                
                # Store the thresholds for plotting
                self.predictions[self.data.index[i]] = {
                    'expected_price': expected_price,
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold
                }

                # Get current price for this period
                current_price = self.data['close'].iloc[i]

                # Generate buy/sell/hold signal based on thresholds
                if current_price <= buy_threshold:
                    self.data.at[self.data.index[i], 'signal'] = 1  # Buy signal
                elif current_price >= sell_threshold:
                    self.data.at[self.data.index[i], 'signal'] = -1  # Sell signal
                else:
                    self.data.at[self.data.index[i], 'signal'] = 0  # Hold

        # Adding position tracking (e.g., holding or neutral)
        self.data['position'] = self.data['signal'].shift(1).fillna(0)

        return self.data
    def plot_indicators(self):
        """
        Plot the actual price along with buy and sell thresholds.
        """
        plt.plot(self.data.index, self.data['close'], label="Actual Price", color="blue")

        # Plot expected price, buy, and sell thresholds using stored predictions
        expected_prices = [self.predictions[idx]['expected_price'] if idx in self.predictions else np.nan for idx in self.data.index]
        buy_thresholds = [self.predictions[idx]['buy_threshold'] if idx in self.predictions else np.nan for idx in self.data.index]
        sell_thresholds = [self.predictions[idx]['sell_threshold'] if idx in self.predictions else np.nan for idx in self.data.index]
        
        plt.plot(self.data.index, expected_prices, label="Expected Price", color="purple", linestyle="--")
        plt.plot(self.data.index, buy_thresholds, label="Buy Threshold", color="green", linestyle="--")
        plt.plot(self.data.index, sell_thresholds, label="Sell Threshold", color="red", linestyle="--")

    def forecast(self):
        """
        Fit the model to self.data and forecast the next 'self.time_periods' periods.
        """
        model = gbm.GBMModel(self.data)
        model.fit()
        start_price = self.data['close'].iloc[-1]

        # Generate forecast dates based on the frequency of the original data
        first_date = self.data.index[0]
        last_date = self.data.index[-1]
        frequency = self.data.index[-1] - self.data.index[-2]

        # if the frequency is a trading day, use bdate_range to exclude weekends
        if frequency == pd.Timedelta('1 days'):
            frequency = 'B'
        else:
            frequency = '15T'  # 15 minutes

        # timedelta to string
        frequency_str = str(frequency)
        # replace '0 days' with ''
        frequency_str = frequency_str.replace('0 days ', '')
        # replace '00:00:00' with ''
        frequency_str = frequency_str.replace('00:00:00', '')

        # durationstr
        duration_str = str(first_date.date()) + '_' + str(last_date.date()) 

        # Simulate future prices
        simulations = model.simulate_future_prices(start_price, frequency,self.time_periods,  self.num_simulations)
        mean, std = np.mean(simulations, axis=0), np.std(simulations, axis=0)

        # show the distribution of the last period
        plt.hist(simulations[:, -1], bins=50, color='blue', alpha=0.5, density=True)
        plt.axvline(mean[-1], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(mean[-1] - 2 * std[-1], color='red', linestyle='dotted', linewidth=2)
        plt.axvline(mean[-1] + 2 * std[-1], color='red', linestyle='dotted', linewidth=2)
        plt.title(f"GBM Forecast Distribution: {self.contract.symbol}")
        plt.xlabel("Price")
        plt.ylabel("Density")
        plt.legend(["Mean", "Mean - 2*Std", "Mean + 2*Std"])
        plt.savefig(f"output/{self.contract.symbol}_forecast_distribution_{frequency_str}_{duration_str}.png")
        plt.clf()



        forecast_dates = pd.bdate_range(start=last_date, periods=self.time_periods + 1, freq=frequency)[1:]
        # Plotting the mean forecast with confidence interval
        plt.plot(forecast_dates, mean, color='blue', label='Mean Forecast')
        plt.fill_between(forecast_dates, mean - 2 * std, mean + 2 * std, color='gray', alpha=0.2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.title(f"GBM Forecast: {self.contract.symbol}")
        plt.xlabel("Date (MM-DD HH:MM)")
        plt.ylabel("Price")
        plt.legend()


        plt.savefig(f"output/{self.contract.symbol}_forecast_{frequency_str}_{duration_str}.png")
        # clear the plot
        plt.clf()

        # Calculate 95% confidence interval
        ci_lower = mean[-1] - 2 * std[-1]
        ci_upper = mean[-1] + 2 * std[-1]

        # Logging forecasted prices with dates
        with open(f"output/{self.contract.symbol}_forecast_{frequency_str}_{duration_str}.txt", "w") as f:
            f.write("Forecasted Prices (Date, Mean, and Std):\n")
            for i, date in enumerate(forecast_dates):
                f.write(f"{date.strftime('%Y-%m-%d %H:%M')}: Mean = {mean[i]:.2f}, Std = {std[i]:.2f}\n")
            f.write("\n95% Confidence Interval:\n")
            f.write(f"Lower Bound: {ci_lower:.2f}\n")
            f.write(f"Upper Bound: {ci_upper:.2f}\n")

            # how many simulations are within the 95% confidence interval
            num_within_ci = np.sum((simulations[:, -1] >= ci_lower) & (simulations[:, -1] <= ci_upper))
            coverage_prob = num_within_ci / self.num_simulations
            f.write(f"\nCoverage Probability: {coverage_prob:.2f}")
            num_higher = np.sum(simulations[:, -1] > ci_upper)
            num_lower = np.sum(simulations[:, -1] < ci_lower)
            f.write(f"\nNumber of Simulations Above Upper Bound: {num_higher}")
            f.write(f"\nNumber of Simulations Below Lower Bound: {num_lower}")

        
# introduce GBM with Volume data
