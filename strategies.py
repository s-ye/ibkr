# strategies.py
from base_strategy import BaseStrategy
import pandas as pd
import matplotlib.pyplot as plt

class SmaCrossoverStrategy(BaseStrategy):
    def __init__(self, contract, data, ib, params):
        super().__init__(contract, data, ib, params)
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
        plt.plot(self.data_with_signals['fast_sma'], label=f"{self.params.get('fast_period', 10)}-Period SMA")
        plt.plot(self.data_with_signals['slow_sma'], label=f"{self.params.get('slow_period', 30)}-Period SMA")



class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, contract, data, ib, params):
        super().__init__(contract, data, ib, params)
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
        plt.plot(self.data_with_signals['sma'], label="SMA")
        plt.plot(self.data_with_signals['upper_band'], label="Upper Band", linestyle='--')
        plt.plot(self.data_with_signals['lower_band'], label="Lower Band", linestyle='--')
