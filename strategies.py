# strategies.py
from base_strategy import BaseStrategy
import pandas as pd
import matplotlib.pyplot as plt
from stoploss_takeprofit_strategy import StopLossTakeProfitStrategy
from datetime import time


class SmaCrossoverStrategy(StopLossTakeProfitStrategy):
    def __init__(self, contract, data, ib, params, initial_capital=1_000_000, position_size_pct=0.02,
                 profit_target_pct=0.05, trailing_stop_pct=0.03):
        super().__init__(contract, data, ib, params, initial_capital=initial_capital,
                         profit_target_pct=profit_target_pct, trailing_stop_pct=trailing_stop_pct)

    def generate_signals(self):
        """Generate buy/sell signals based on SMA crossover."""
        self.data['fast_sma'] = self.data['close'].rolling(window=self.params.get('fast_period', 10)).mean()
        self.data['slow_sma'] = self.data['close'].rolling(window=self.params.get('slow_period', 30)).mean()

        # Generate signals: 1 for Buy, -1 for Sell, 0 for Neutral
        self.data['signal'] = 0
        self.data.loc[self.data['fast_sma'] > self.data['slow_sma'], 'signal'] = 1
        self.data.loc[self.data['fast_sma'] < self.data['slow_sma'], 'signal'] = -1
        self.data['position'] = self.data['signal'].shift(1)
        
        return self.data
    
    def plot_indicators(self):
        """Plot the SMA lines."""
        plt.plot(self.data_with_signals['fast_sma'], label=f"{self.params.get('fast_period', 10)}-Period SMA", color='orange')
        plt.plot(self.data_with_signals['slow_sma'], label=f"{self.params.get('slow_period', 30)}-Period SMA", color='purple')

class BollingerBandsStrategy(StopLossTakeProfitStrategy):
    def __init__(self, contract, data, ib, params, initial_capital=1_000_000, position_size_pct=0.02,
                 profit_target_pct=0.05, trailing_stop_pct=0.03):
        super().__init__(contract, data, ib, params, initial_capital=initial_capital,
                         profit_target_pct=profit_target_pct, trailing_stop_pct=trailing_stop_pct)

    def generate_signals(self):
        """Generate buy/sell signals based on Bollinger Bands."""
        window = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2)

        self.data['sma'] = self.data['close'].rolling(window=window).mean()
        self.data['upper_band'] = self.data['sma'] + std_dev * self.data['close'].rolling(window=window).std()
        self.data['lower_band'] = self.data['sma'] - std_dev * self.data['close'].rolling(window=window).std()

        # Generate signals: 1 for Buy when price crosses above lower band, -1 for Sell when it crosses below upper band
        self.data['signal'] = 0
        self.data.loc[self.data['close'] < self.data['lower_band'], 'signal'] = 1
        self.data.loc[self.data['close'] > self.data['upper_band'], 'signal'] = -1
        self.data['position'] = self.data['signal'].shift(1)
        
        return self.data

    def plot_indicators(self):
        """Plot Bollinger Bands."""
        plt.plot(self.data_with_signals['sma'], label="SMA", color="purple", linestyle="--")
        plt.plot(self.data_with_signals['upper_band'], label="Upper Band", color="orange", linestyle="--")
        plt.plot(self.data_with_signals['lower_band'], label="Lower Band", color="orange", linestyle="--")


# My concern is that the bollinger bands strategy is not that effective if the stock does not go up.
# I want to create a sideways bollinger bands strategy that will work better in sideways markets.
# sideways_bollinger_bands_strategy.py

class SidewaysBollingerBandsStrategy(StopLossTakeProfitStrategy):

    def __init__(self, contract, data, ib, params, initial_capital=1_000_000, position_size_pct=0.02,
                 profit_target_pct=0.05, trailing_stop_pct=0.03):
        super().__init__(contract, data, ib, params, initial_capital=initial_capital,
                         profit_target_pct=profit_target_pct, trailing_stop_pct=trailing_stop_pct)
        self.rsi_window = self.params.get('rsi_window', 14)

    def generate_signals(self):
        """Generate buy/sell signals based on Bollinger Bands and set stop-loss/take-profit."""
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2)
        self.data['sma'] = self.data['close'].rolling(window=period).mean()
        self.data['upper_band'] = self.data['sma'] + std_dev * self.data['close'].rolling(window=period).std()
        self.data['lower_band'] = self.data['sma'] - std_dev * self.data['close'].rolling(window=period).std()

        # calculate rsi
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

        self.data['signal'] = 0
        self.data.loc[self.data['close'] < self.data['lower_band'], 'signal'] = 1  # Buy at lower band
        self.data.loc[self.data['close'] > self.data['sma'], 'signal'] = -1  # Sell at middle band

        return self.data

    def _should_buy(self, i):
        """Custom buy logic based on Sideways Bollinger Bands."""
        return self.data_with_signals['signal'].iloc[i] == 1 and self.data_with_signals['signal'].iloc[i - 1] != 1 and self.data_with_signals['rsi'].iloc[i] < 30
    
class DipRecoverVolumeStrategy(StopLossTakeProfitStrategy):
    def __init__(self, stock, data, ib, params=None, initial_capital=1_000_000, position_size_pct=0.02, profit_target_pct=0.05, trailing_stop_pct=0.03):
        super().__init__(stock, data, ib, params, initial_capital, position_size_pct, profit_target_pct, trailing_stop_pct)

    def generate_signals(self):
        """
        Generate buy/sell signals based on the dip-recovery pattern.
        A 'dip' is identified by a price drop in the morning (e.g., before 11 AM),
        followed by a recovery after noon.
        """
        self.data['signal'] = 0  # Reset signals
        std_dev = self.params.get('std_dev', 1.5)
        period = self.params.get('period', 10)


        # Calculate rolling mean and standard deviation to detect morning dips
        self.data['rolling_mean'] = self.data['close'].rolling(window=period).mean()
        self.data['rolling_std'] = self.data['close'].rolling(window=period).std()
        self.data['rolling_vol'] = self.data['volume'].rolling(window=period).mean()

        # calculate rsi
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))



        self.data['is_dip'] = ((self.data['close'] < self.data['rolling_mean'] - std_dev * self.data['rolling_std']) & (self.data['volume'] > self.data['rolling_vol']))

        # Generate buy signals: Buy on dip with volume confirmation
        self.data.loc[self.data['is_dip'], 'signal'] = 1

        # Generate sell signals: Sell when RSI is above 70 and price is above rolling mean and volume is above rolling volume
        self.data.loc[(self.data['rsi'] > 70) & (self.data['close'] > self.data['rolling_mean']) & (self.data['volume'] > self.data['rolling_vol']), 'signal'] = -1
        # show the buy and sell signals
        return self.data

    def _should_buy(self, i):
        return self.data['signal'].iloc[i] == 1 and self.data['signal'].iloc[i - 1] != 1

    def _should_sell(self, i):
        return self.data['signal'].iloc[i] == -1 and self.data['signal'].iloc[i - 1] != -1
    
    def run_strategy(self):
        """
        Execute the strategy, which buys during morning dips and sells on afternoon recovery,
        with stop-loss and take-profit in place from the superclass.
        """
        self._execute_trades()

    def plot_indicators(self):
        """
        Override to plot morning dip and afternoon recovery indicators for analysis.
        """
        plt.plot(self.data_with_signals['rolling_mean'], label='Rolling Mean', linestyle='--', color='orange')
        # show the rolling mean - std dev * std dev
        plt.plot(self.data_with_signals['rolling_mean'] - self.params.get('std_dev', 1.5) * self.data_with_signals['rolling_std'], label='Upper Band', linestyle='--', color='red')
        # show the rolling mean + std dev * std dev
        plt.plot(self.data_with_signals['rolling_mean'] + self.params.get('std_dev', 1.5) * self.data_with_signals['rolling_std'], label='Lower Band', linestyle='--', color='green')
        # show the volume as a bar chart with seperate y axis


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
