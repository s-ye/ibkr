# main.py
from ib_insync import IB, Stock, util
from strategies import *
from itertools import product

class Backtester:
    def __init__(self, stock_symbol, exchange, currency, client_id=1):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=client_id)
        self.stock = Stock(stock_symbol, exchange, currency)
        self.ib.qualifyContracts(self.stock)
        self.data = self._get_historical_data()

    def _get_historical_data(self):
        bars = self.ib.reqHistoricalData(
            self.stock,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 min',
            whatToShow='MIDPOINT',
            useRTH=True
        )
        return util.df(bars)

    def run_sma_strategy(self, sma_params):
        for fast_period, slow_period, take_profit_pct, stop_loss_pct in product(
                sma_params['fast_period'],
                sma_params['slow_period'],
                sma_params['take_profit_pct'],
                sma_params['stop_loss_pct']
            ):
            print(f"\nRunning SMA Crossover Strategy with fast_period={fast_period}, slow_period={slow_period}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            sma_strategy = SmaCrossoverStrategy(
                self.stock, self.data, self.ib,
                params={'fast_period': fast_period, 'slow_period': slow_period},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            sma_results = sma_strategy.backtest()
            sma_final_portfolio_value = sma_strategy.final_portfolio_value  # Get the final portfolio value after backtest
            sma_strategy.plot_trades()
            print(sma_strategy.trade_statistics())

    def run_bb_strategy(self, bb_params):
        for period, std_dev, take_profit_pct, stop_loss_pct in product(
                bb_params['period'],
                bb_params['std_dev'],
                bb_params['take_profit_pct'],
                bb_params['stop_loss_pct']
            ):
            print(f"\nRunning Bollinger Bands Strategy with period={period}, std_dev={std_dev}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            bb_strategy = BollingerBandsStrategy(
                self.stock, self.data, self.ib,
                params={'period': period, 'std_dev': std_dev},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            bb_results = bb_strategy.backtest()
            bb_final_portfolio_value = bb_strategy.final_portfolio_value  # Get the final portfolio value after backtest
            bb_strategy.plot_trades()
            print(bb_strategy.trade_statistics())

    def run_sbb_strategy(self, sbb_params):
        for period, std_dev, rsi_window, take_profit_pct, stop_loss_pct in product(
                sbb_params['period'],
                sbb_params['std_dev'],
                sbb_params['rsi_window'],
                sbb_params['take_profit_pct'],
                sbb_params['stop_loss_pct']
            ):
            print(f"\nRunning Sideways Bollinger Bands Strategy with period={period}, std_dev={std_dev}, rsi_window={rsi_window}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            sbb_strategy = SidewaysBollingerBandsStrategy(
                self.stock, self.data, self.ib,
                params={'period': period, 'std_dev': std_dev, 'rsi_window': rsi_window},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            sbb_results = sbb_strategy.backtest()
            sbb_final_portfolio_value = sbb_strategy.final_portfolio_value  # Get the final portfolio value after backtest
            sbb_strategy.plot_trades()
            print(sbb_strategy.trade_statistics())

    def disconnect(self):
        self.ib.disconnect()

# Example usage
if __name__ == "__main__":
    sma_params = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 40],
        'take_profit_pct': [0.05, 0.1],  # Profit target as a percentage
        'stop_loss_pct': [0.03, 0.05]    # Stop loss as a percentage
    }

    bb_params = {
        'period': [15, 20, 25],
        'std_dev': [1, 1.5, 2, 2.5],
        'take_profit_pct': [0.05, 0.1],  # Profit target as a percentage
        'stop_loss_pct': [0.03, 0.05]    # Stop loss as a percentage
    }

    sbb_params = {
        'period': [15, 20, 25],
        'std_dev': [1, 1.5, 2, 2.5],
        'rsi_window': [5, 14],
        'take_profit_pct': [0.05, 0.1],  # Profit target as a percentage
        'stop_loss_pct': [0.03, 0.05]    # Stop loss as a percentage
    }

    backtester = Backtester('CVNA', 'SMART', 'USD')
    backtester.run_sma_strategy(sma_params)
    # backtester.run_bb_strategy(bb_params)
    # backtester.run_sbb_strategy(sbb_params)
    backtester.disconnect()