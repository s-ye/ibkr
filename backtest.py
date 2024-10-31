# main.py
from ib_insync import IB, Stock, util
from strategies import *
from itertools import product

# Define ranges of hyperparameters to test for each strategy
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

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define stock and parameters
stock = Stock('NVO', 'SMART', 'USD')
ib.qualifyContracts(stock)

# Request historical data (15 minute delay)
bars = ib.reqHistoricalData(
    stock,
    endDateTime='',
    durationStr='1 Y',
    barSizeSetting='15 mins',
    whatToShow='MIDPOINT',
    useRTH=True
)
data = util.df(bars)

# Run SMA Crossover Strategy with different hyperparameters
for fast_period, slow_period, take_profit_pct, stop_loss_pct in product(
        sma_params['fast_period'],
        sma_params['slow_period'],
        sma_params['take_profit_pct'],
        sma_params['stop_loss_pct']
    ):
    print(f"\nRunning SMA Crossover Strategy with fast_period={fast_period}, slow_period={slow_period}, "
          f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
    sma_strategy = SmaCrossoverStrategy(
        stock, data, ib,
        params={'fast_period': fast_period, 'slow_period': slow_period},
        initial_capital=1_000_000,
        profit_target_pct=take_profit_pct,
        trailing_stop_pct=stop_loss_pct
    )
    sma_results = sma_strategy.backtest()
    sma_final_portfolio_value = sma_strategy.final_portfolio_value  # Get the final portfolio value after backtest
    sma_strategy.plot_trades()
    print(sma_strategy.trade_statistics())

# Run Bollinger Bands Strategy with different hyperparameters
for period, std_dev, take_profit_pct, stop_loss_pct in product(
        bb_params['period'],
        bb_params['std_dev'],
        bb_params['take_profit_pct'],
        bb_params['stop_loss_pct']
    ):
    print(f"\nRunning Bollinger Bands Strategy with period={period}, std_dev={std_dev}, "
          f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
    bb_strategy = BollingerBandsStrategy(
        stock, data, ib,
        params={'period': period, 'std_dev': std_dev},
        initial_capital=1_000_000,
        profit_target_pct=take_profit_pct,
        trailing_stop_pct=stop_loss_pct
    )
    bb_results = bb_strategy.backtest()
    bb_final_portfolio_value = bb_strategy.final_portfolio_value  # Get the final portfolio value after backtest
    bb_strategy.plot_trades()
    print(bb_strategy.trade_statistics())

# Run Sideways Bollinger Bands Strategy with different hyperparameters
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
        stock, data, ib,
        params={'period': period, 'std_dev': std_dev, 'rsi_window': rsi_window},
        initial_capital=1_000_000,
        profit_target_pct=take_profit_pct,
        trailing_stop_pct=stop_loss_pct
    )
    sbb_results = sbb_strategy.backtest()
    sbb_final_portfolio_value = sbb_strategy.final_portfolio_value  # Get the final portfolio value after backtest
    sbb_strategy.plot_trades()
    print(sbb_strategy.trade_statistics())

ib.disconnect()