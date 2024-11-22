# strategies.py
from base_strategy import BaseStrategy
import pandas as pd
import matplotlib.pyplot as plt
from stoploss_takeprofit_strategy import StopLossTakeProfitStrategy
import matplotlib.dates as mdates
from datetime import time

import numpy as np
import matplotlib.pyplot as plt
import gbm as gbm

class GBMStrategy(StopLossTakeProfitStrategy):
    # historical_data is the data used to fit the model
    # data is the data used to generate signals
    # the model will also be updated as new data comes in
    def __init__(self, contract, data, historical_data,
                 ib, params,initial_capital=1_000_000, position_size_pct=0.02,profit_target_pct=0.05, trailing_stop_pct = 0.03):
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
        num_periods = len(self.data)
        predictions = {}  # Dictionary to store predictions for each time_periods block
        min_data_points = self.time_periods  # Minimum data points required to fit the model

        # consider the historical data
        # we initialize the model with the historical data up to the current period
        # and then generate predictions for the next 'self.time_periods' periods

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
            simulations = model.simulate_future_prices(start_price, self.time_periods, self.num_simulations)
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

        plt.title(f"GBM Strategy: {self.contract.symbol}")
        plt.legend()
        plt.show()


    def forecast(self):
        """
        Fit the model to self.data and forecast the next 'self.time_periods' periods.
        """
        model = gbm.GBMModel(self.data)
        model.fit()
        start_price = self.data['close'].iloc[-1]

        # Simulate future prices
        simulations = model.simulate_future_prices(start_price, self.time_periods, self.num_simulations)
        mean, std = np.mean(simulations, axis=0), np.std(simulations, axis=0)

        # Generate forecast dates based on the frequency of the original data
        last_date = self.data.index[-1]
        frequency = self.data.index[-1] - self.data.index[-2]
        forecast_dates = pd.bdate_range(start=last_date, periods=self.time_periods + 1)[1:]

        # Plotting the mean forecast with confidence interval
        plt.plot(forecast_dates, mean, color='blue', label='Mean Forecast')
        plt.fill_between(forecast_dates, mean - 2 * std, mean + 2 * std, color='gray', alpha=0.2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.title(f"GBM Forecast: {self.contract.symbol}")
        plt.xlabel("Date (MM-DD)")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(f"output/{self.contract.symbol}_forecast.png")

        # Calculate 95% confidence interval
        ci_lower = mean[-1] - 2 * std[-1]
        ci_upper = mean[-1] + 2 * std[-1]

        # Logging forecasted prices with dates
        with open(f"output/{self.contract.symbol}_forecast.txt", "w") as f:
            f.write("Forecasted Prices (Date, Mean, and Std):\n")
            for i, date in enumerate(forecast_dates):
                f.write(f"{date.strftime('%Y-%m-%d %H:%M')}: Mean = {mean[i]:.2f}, Std = {std[i]:.2f}\n")
            f.write("\n95% Confidence Interval:\n")
            f.write(f"Lower Bound: {ci_lower:.2f}\n")
            f.write(f"Upper Bound: {ci_upper:.2f}\n")





    

        

# introduce GBM with Volume data
