# strategies.py
from base_strategy import BaseStrategy
import pandas as pd
import matplotlib.pyplot as plt
from stoploss_takeprofit_strategy import StopLossTakeProfitStrategy
from datetime import time

import numpy as np
import matplotlib.pyplot as plt
import gbm as gbm

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
        num_periods = len(self.data)
        predictions = {}  # Dictionary to store predictions for each time_periods block
        min_data_points = self.time_periods  # Minimum data points required to fit the model

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
