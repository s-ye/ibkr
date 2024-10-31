import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, Stock, util
from datetime import datetime
import logging
import os

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseStrategy:
    def __init__(self, stock, data, ib, params=None, initial_capital=1_000_000, position_size_pct=0.02):
        self.stock = stock
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

    def log_statistics_and_trades(self, statistics):
        # Format hyperparameters in filename
        hyperparam_str = "_".join([f"{k}{v}" for k, v in self.params.items()])
        log_filename = os.path.join("output", f"{self.__class__.__name__}_{hyperparam_str}.log")
        
        # Prepare statistics string
        stats_str = f"{datetime.now()} - Final statistics: "
        for key, value in statistics.items():
            stats_str += f"{key}: {value}, " + "\n"
        stats_str += "\n\n"

        # If the file already exists, read its current contents
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as file:
                existing_content = file.read()
        else:
            existing_content = ""

        # Write the stats at the top of the log file
        with open(log_filename, 'w') as file:
            file.write(stats_str)
            file.write(existing_content)

    def _setup_logger(self):
        """Sets up a logger for the strategy instance that overwrites on each run."""
        # Format the hyperparameters for file naming
        hyperparam_str = "_".join([f"{k}{v}" for k, v in self.params.items()])
        log_filename = os.path.join("output", f"{self.__class__.__name__}_{hyperparam_str}.log")

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{hyperparam_str}")

        # Clear existing handlers to avoid duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create file handler in overwrite mode
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        # Log the starting conditions
        self.logger.info(f"Starting {self.__class__.__name__} with initial capital: {self.initial_capital}")
        self.logger.info(f"Hyperparameters: {self.params}")
    
    def generate_signals(self):
        raise NotImplementedError("generate_signals method must be implemented in subclasses.")

    def backtest(self):
        # Prepare data and index for backtesting
        if 'date' not in self.data.columns:
            self.data.index = pd.to_datetime(self.data.index, unit='s')
            self.data['date'] = self.data.index
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data.set_index('date', inplace=True)
        self.data.dropna(inplace=True)

        # Generate signals and execute trades based on them
        self.data_with_signals = self.generate_signals()
        self._execute_trades()
        
        # After all trades, store the final portfolio value
        self.final_portfolio_value = self.current_balance + (self.current_position * self.data_with_signals['close'].iloc[-1])
        return self.data_with_signals

    # other methods remain the same

    def _execute_trades(self):
        self.current_position = 0  # Initialize total shares currently held
        self.current_balance = self.initial_capital  # Set to initial capital
        self.avg_entry_price = 0  # Reset average entry price for new backtest run
        self.entry_date = None  # Reset entry date for new position

        for i in range(1, len(self.data_with_signals)):
            current_price = self.data_with_signals['close'].iloc[i]
            # Calculate portfolio value (cash balance + value of held shares)
            portfolio_value = self.current_balance + (self.current_position * current_price)
            
            # Track portfolio value over time
            self.portfolio_values.append({
                'date': self.data_with_signals.index[i],
                'portfolio_value': portfolio_value
            })

            # Buy on signal change to 1 (if the previous signal was not 1)
            if self.data_with_signals['signal'].iloc[i] == 1 and self.data_with_signals['signal'].iloc[i-1] != 1:
                shares_to_buy = (self.current_balance * self.position_size_pct) / current_price
                buy_cost = shares_to_buy * current_price
                if shares_to_buy > 0 and buy_cost <= self.current_balance:
                    # Update average entry price and total position size
                    self.avg_entry_price = ((self.avg_entry_price * self.current_position) + (current_price * shares_to_buy)) / (self.current_position + shares_to_buy)
                    self.current_balance -= buy_cost
                    self.current_position += shares_to_buy  # Update the current position with new shares
                    self.entry_date = self.data_with_signals.index[i]  # Set entry date for holding duration
                    print(f"Buy {shares_to_buy} shares at {current_price} on {self.entry_date}")
                    # write to log file
                    self.logger.info(f"Buy {shares_to_buy} shares at {current_price} on {self.entry_date}")

            # Sell on signal change to -1 (if the previous signal was 1)
            elif self.data_with_signals['signal'].iloc[i] == -1 and self.data_with_signals['signal'].iloc[i-1] != -1:
                if self.current_position > 0 and self.entry_date is not None:
                    sell_revenue = self.current_position * current_price
                    profit = (current_price - self.avg_entry_price) * self.current_position
                    duration_minutes = (self.data_with_signals.index[i] - self.entry_date).total_seconds() / 60  # Duration in minutes
                    self.current_balance += sell_revenue  # Add revenue from the sale to the balance
                    self.trades.append({
                        'entry_date': self.entry_date,
                        'exit_date': self.data_with_signals.index[i],
                        'entry_price': self.avg_entry_price,
                        'exit_price': current_price,
                        'shares': self.current_position,
                        'return': profit / (self.avg_entry_price * self.current_position),
                        'profit': profit,
                        'duration': duration_minutes  # Store duration in minutes
                    })
                    print(f"Sell {self.current_position} shares at {current_price} on {self.data_with_signals.index[i]} (Profit: {profit}, Duration: {duration_minutes} minutes)")
                    # write to log file
                    self.logger.info(f"Sell {self.current_position} shares at {current_price} on {self.data_with_signals.index[i]} (Profit: {profit}, Duration: {duration_minutes} minutes)")
                    # Reset position tracking after selling
                    self.current_position = 0
                    self.avg_entry_price = 0
                    self.entry_date = None  # Reset entry date after a full position is closed

        # Final portfolio value calculation if still holding shares
        if self.current_position > 0:
            final_price = self.data_with_signals['close'].iloc[-1]
            self.final_portfolio_value = self.current_balance + (self.current_position * final_price)
        else:
            self.final_portfolio_value = self.current_balance

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
        
        # Calculate trade statistics
        stats = {
            'total_trades': len(df_trades),
            'winning_trades': df_trades[df_trades['return'] > 0].shape[0],
            'losing_trades': df_trades[df_trades['return'] <= 0].shape[0],
            'average_return': df_trades['return'].mean(),
            'average_duration': df_trades['duration'].mean(),
            'final_balance': self.current_balance,
            'final_portfolio_value': self.final_portfolio_value
        }

        self.log_statistics_and_trades(stats)

    def plot_trades(self, filename=None):
        """Plot the close price with indicators and buy/sell signals for the specified strategy and save to a file."""
        # Set up the plot
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_with_signals['close'], label='Close Price', color='blue')
        self.plot_indicators()

        # Plot buy and sell signals based on signal changes
        buys = self.data_with_signals[(self.data_with_signals['signal'] == 1) & (self.data_with_signals['signal'].shift(1) != 1)]
        buy_dates = buys.index
        buy_prices = buys['close']

        sells = self.data_with_signals[(self.data_with_signals['signal'] == -1) & (self.data_with_signals['signal'].shift(1) != 1)]
        sell_dates = sells.index
        sell_prices = sells['close']

        if not buy_dates.empty:
            plt.plot(buy_dates, buy_prices, '^', markersize=10, color='green', label='Buy Signal')
        if not sell_dates.empty:
            plt.plot(sell_dates, sell_prices, 'v', markersize=10, color='red', label='Sell Signal')

        # Add title and labels
        param_str = "_".join([f"{k}{v}" for k, v in self.params.items()])
        plt.title(f"{self.stock.symbol} - {self.__class__.__name__} Strategy ({param_str})")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        final_value_str = f"{self.final_portfolio_value:.2f}" if self.final_portfolio_value is not None else "NA"
        filename = f"output/{final_value_str}_{self.__class__.__name__}_{param_str}.png"

        # Save plot to the file
        plt.savefig(filename)
        plt.close()

    def plot_indicators(self):
        """Override in subclasses to plot strategy-specific indicators."""
        pass