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
        if not self.trades and self.current_position == 0:
            # No trades and no open position
            print("No trades were recorded.")
            # write to log file
            self.logger.info("No trades were recorded.")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'average_return': None,
                'average_duration': None,
                'final_balance': self.current_balance,
                'final_portfolio_value': self.final_portfolio_value
            }
        
        # Convert trades to DataFrame if trades were recorded
        df_trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=['return', 'duration'])
        
        # Check for open position at the end of the backtest
        if self.current_position > 0:
            final_price = self.data_with_signals['close'].iloc[-1]
            unrealized_return = (final_price - self.avg_entry_price) / self.avg_entry_price
            unrealized_duration = (self.data_with_signals.index[-1] - self.entry_date).total_seconds() / 60  # Duration in minutes
            
            # Add hypothetical trade to the DataFrame
            df_trades = pd.concat([df_trades, pd.DataFrame([{
                'entry_date': self.entry_date,
                'exit_date': self.data_with_signals.index[-1],
                'entry_price': self.avg_entry_price,
                'exit_price': final_price,
                'shares': self.current_position,
                'return': unrealized_return,
                'profit': unrealized_return * self.avg_entry_price * self.current_position,
                'duration': unrealized_duration
            }])], ignore_index=True)
            
            print(f"Holding {self.current_position} shares as an open position at the end with unrealized return: {unrealized_return}, Duration: {unrealized_duration} minutes")
            # write to log file
            self.logger.info(f"Holding {self.current_position} shares as an open position at the end with unrealized return: {unrealized_return}, Duration: {unrealized_duration} minutes")
        
        # Force columns to numeric for calculations
        df_trades['return'] = pd.to_numeric(df_trades['return'], errors='coerce')
        df_trades['duration'] = pd.to_numeric(df_trades['duration'], errors='coerce')
        
        def calculate_sharpe(df_trades):
            """
            Calculate the Sharpe Ratio from trade returns, assuming duration is in minutes.
            """
            # Filter out trades with very short durations (e.g., <1 minute) to avoid division by very small numbers
            df_trades = df_trades[df_trades['duration'] > 1]

            # Calculate daily returns, making sure 'duration' is in minutes
            df_trades['daily_return'] = df_trades['return'] / (df_trades['duration'] / (60 * 24))
            daily_returns = df_trades['daily_return']

            # Debugging: Print mean and standard deviation of daily returns
            print(f"Mean daily return: {daily_returns.mean()}")
            print(f"Standard deviation of daily returns: {daily_returns.std()}")

            # Calculate Sharpe Ratio, using np.sqrt(252) to annualize
            if daily_returns.std() != 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                print("Standard deviation of daily returns is zero, setting Sharpe Ratio to 0.")
                sharpe_ratio = 0  # Avoid division by zero if standard deviation is zero

            return sharpe_ratio


        self.sharpe_ratio = calculate_sharpe(df_trades)
        # Calculate trade statistics
        stats = {
            'Stock': self.stock.symbol,
            'total_trades': len(df_trades),
            'winning_trades': df_trades[df_trades['return'] > 0].shape[0],
            'losing_trades': df_trades[df_trades['return'] <= 0].shape[0],
            'average_return': df_trades['return'].mean(),
            'average_duration': df_trades['duration'].mean(),
            'final_balance': self.current_balance,
            'final_portfolio_value': self.final_portfolio_value,
            'sharpe_ratio': self.sharpe_ratio
        }

        self.log_statistics_and_trades(stats)
        return stats

    def plot_trades(self, filename=None):
        pass

    def plot_indicators(self):
        """Override in subclasses to plot strategy-specific indicators."""
        pass