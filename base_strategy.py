import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, Stock, util
from datetime import datetime
import logging
import os
import matplotlib.dates as mdates

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseStrategy:
    def __init__(self, stock, data, ib, params=None, initial_capital=1_000_000, position_size_pct=0.02):
        self.stock = stock
        data = data.copy()
        self.data = data  # Historical data

        self.ib = ib  # IBKR connection
        self.params = params if params else {}
        self.initial_capital = initial_capital  # Starting capital
        self.current_balance = initial_capital  # Current balance for trade calculations
        self.position_size_pct = position_size_pct  # Percent of balance to allocate per trade
        self.trades = []  # Track trades
        self.data_with_signals = None  # Store data with signals
        self.current_position = 0  # Tracks the number of shares currently held
        self.portfolio_values = []  # Store portfolio value over time
        self.avg_entry_price = 0  # Track average entry price
        self.entry_date = None  # Track the date when the current position was opened
        self.stats = None  # Store trade statistics
        self.final_portfolio_value = None
        self.sharpe_ratio = None

    def log_statistics_and_trades(self, statistics):
        pass

    def _setup_logger(self):
        """Sets up a logger for the strategy instance that overwrites on each run."""
        pass
    
    def generate_signals(self):
        raise NotImplementedError("generate_signals method must be implemented in subclasses.")

    def backtest(self):

    # Prepare data and index for backtesting
        if 'date' not in self.data.columns:
            # Convert index to datetime with UTC
            self.data.index = pd.to_datetime(self.data.index, utc=True)
            self.data['date'] = self.data.index
        else:
            # Ensure 'date' column is in datetime format with UTC
            self.data['date'] = pd.to_datetime(self.data['date'], utc=True, errors='coerce')
        
        # Set 'date' column as the index for the DataFrame
        self.data.set_index('date', inplace=True)
        self.data.dropna(inplace=True)  # Drop rows where datetime conversion failed
        
        # Generate signals and execute trades based on them
        self.data_with_signals = self.generate_signals()
        self._execute_trades()
        
        # After all trades, store the final portfolio value, if trades were made
        if self.trades:
            self.final_portfolio_value = self.current_balance + (self.current_position * self.data_with_signals['close'].iloc[-1])
        else:
            self.final_portfolio_value = self.current_balance
        return self.data_with_signals

    # other methods remain the same

    def _execute_trades(self):
       raise NotImplementedError("_execute_trades method must be implemented in subclasses.")

    def trade_statistics(self):
        pass

    def plot_trades(self, filename=None):
        pass

    def plot_indicators(self):
        """Override in subclasses to plot strategy-specific indicators."""
        pass