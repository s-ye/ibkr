# stop_loss_take_profit_strategy.py
# stop_loss_take_profit_strategy.py
import matplotlib.pyplot as plt
from base_strategy import BaseStrategy
import logging
import os

class StopLossTakeProfitStrategy(BaseStrategy):
    def __init__(self, stock, data, ib, params=None, initial_capital=1_000_000, position_size_pct=0.02, profit_target_pct=0.05, trailing_stop_pct=0.03):
        super().__init__(stock, data, ib, params, initial_capital, position_size_pct)
        self.profit_target_pct = profit_target_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.current_position = 0
        self.current_balance = self.initial_capital
        self.avg_entry_price = 0
        self.entry_date = None
        self.highest_price_since_entry = 0
        self._setup_logger()

    # existing methods like _execute_trades, _should_buy, etc.

    def plot_trades(self, filename=None):
        """Plot close price with indicators, buy signals, profit-taking, and stop-loss exits."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_with_signals['close'], label='Close Price', color='blue')
        self.plot_indicators()

        # Identify Buy signals
        buys = self.data_with_signals[(self.data_with_signals['signal'] == 1) & (self.data_with_signals['signal'].shift(1) != 1)]
        buy_dates = buys.index
        buy_prices = buys['close']

        # Identify Profit-Taking Exits
        profit_exits = self.data_with_signals[self.data_with_signals['profit_take']]
        profit_dates = profit_exits.index
        profit_prices = profit_exits['close']

        # Identify Stop-Loss Exits
        stop_exits = self.data_with_signals[self.data_with_signals['stop_loss']]
        stop_dates = stop_exits.index
        stop_prices = stop_exits['close']

        # Plot Buy Signals
        if not buy_dates.empty:
            plt.plot(buy_dates, buy_prices, '^', markersize=10, color='green', label='Buy Signal')

        # Plot Profit-Taking Exits
        if not profit_dates.empty:
            plt.plot(profit_dates, profit_prices, 'o', markersize=8, color='gold', label='Profit-Take Exit')

        # Plot Stop-Loss Exits
        if not stop_dates.empty:
            plt.plot(stop_dates, stop_prices, 'x', markersize=8, color='red', label='Stop-Loss Exit')

        # Add title, legend, and labels
        param_str = "_".join([f"{k}{v}" for k, v in self.params.items()])
        # Add profit target and stop loss to title
        param_str += f"_PT{self.profit_target_pct}_SL{self.trailing_stop_pct}"
        plt.title(f"{self.stock.symbol} - {self.__class__.__name__} Strategy ({param_str})")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Set dynamic filename based on final portfolio value
        final_value_str = f"{self.final_portfolio_value:.2f}" if self.final_portfolio_value is not None else "NA"
        filename = filename or f"output/{final_value_str}_{self.__class__.__name__}_{param_str}.png"

        # Save plot to file
        plt.savefig(filename)
        plt.close()

    def plot_indicators(self):
        """Override in subclasses to plot strategy-specific indicators."""
        pass

    def _execute_trades(self):
        """Manage entries and exits with stop-loss and take-profit logic."""
        self.data_with_signals['profit_take'] = False
        self.data_with_signals['stop_loss'] = False

        for i in range(1, len(self.data_with_signals)):
            current_price = self.data_with_signals['close'].iloc[i]
            portfolio_value = self.current_balance + (self.current_position * current_price)
            self.portfolio_values.append({'date': self.data_with_signals.index[i], 'portfolio_value': portfolio_value})

            # Buy logic (to be customized in subclass)
            if self._should_buy(i):
                self._buy_position(current_price, i)

            # Sell logic (profit-taking or stop-loss)
            elif self.current_position > 0:
                if self._hit_profit_target(current_price):
                    self._sell_position(current_price, i, 'profit_take')
                elif self._hit_trailing_stop(current_price):
                    self._sell_position(current_price, i, 'stop_loss')

            # Update highest price since entry if holding position
            if self.current_position > 0:
                self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)

        # Final portfolio value calculation if still holding shares
        self.final_portfolio_value = self.current_balance + (self.current_position * self.data_with_signals['close'].iloc[-1])

    def _should_buy(self, i):
        """Determine whether a buy signal exists. To be customized in subclass."""
        raise NotImplementedError("Subclasses should implement _should_buy.")

    def _buy_position(self, current_price, index):
        """Execute a buy operation."""
        shares_to_buy = (self.current_balance * self.position_size_pct) / current_price
        buy_cost = shares_to_buy * current_price
        if shares_to_buy > 0 and buy_cost <= self.current_balance:
            self.avg_entry_price = ((self.avg_entry_price * self.current_position) + (current_price * shares_to_buy)) / (self.current_position + shares_to_buy)
            self.current_balance -= buy_cost
            self.current_position += shares_to_buy
            self.entry_date = self.data_with_signals.index[index]
            self.highest_price_since_entry = current_price
            self.logger.info(f"Buy {shares_to_buy:.2f} shares at {current_price} on {self.entry_date}")

    def _sell_position(self, current_price, index, exit_type):
        """Helper method to execute a sell operation and log the type of exit."""
        sell_revenue = self.current_position * current_price
        profit = (current_price - self.avg_entry_price) * self.current_position
        duration_minutes = (self.data_with_signals.index[index] - self.entry_date).total_seconds() / 60
        self.current_balance += sell_revenue
        self.trades.append({
            'entry_date': self.entry_date,
            'exit_date': self.data_with_signals.index[index],
            'entry_price': self.avg_entry_price,
            'exit_price': current_price,
            'shares': self.current_position,
            'return': profit / (self.avg_entry_price * self.current_position),
            'profit': profit,
            'duration': duration_minutes
        })
        self.data_with_signals.at[self.data_with_signals.index[index], exit_type] = True
        self.logger.info(f"Sell {self.current_position:.2f} shares at {current_price} on {self.data_with_signals.index[index]} (Exit: {exit_type.capitalize()}, Profit: {profit}, Duration: {duration_minutes} minutes)")
        self.current_position = 0
        self.avg_entry_price = 0
        self.entry_date = None
        self.highest_price_since_entry = 0

    def _hit_profit_target(self, current_price):
        """Check if the profit target is met."""
        return (current_price - self.avg_entry_price) / self.avg_entry_price >= self.profit_target_pct

    def _hit_trailing_stop(self, current_price):
        """Check if the trailing stop is met."""
        if self.highest_price_since_entry == 0:
            return False
        return (self.highest_price_since_entry - current_price) / self.highest_price_since_entry >= self.trailing_stop_pct
    
    def _setup_logger(self):
        """Sets up a logger for the strategy instance that overwrites on each run."""
        # Format the hyperparameters for file naming
        hyperparam_str = "_".join([f"{k}{v}" for k, v in self.params.items()])
        hyperparam_str += f"_PT{self.profit_target_pct}_SL{self.trailing_stop_pct}"
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