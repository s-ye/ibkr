# main.py
from ib_insync import IB, Stock, util
from strategies import SmaCrossoverStrategy, BollingerBandsStrategy

# Define ranges of hyperparameters to test for each strategy
sma_params = {
    'fast_period': [5, 10, 15],
    'slow_period': [20, 30, 40]
}
# looks like bb always performs better than sma and the best hyperparameters are period=25 and std_dev=2
bb_params = {
    'period': [15, 20, 25],
    'std_dev': [1.5, 2, 2.5]
}

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define stock and parameters
stock = Stock('SAVE', 'SMART', 'USD')
ib.qualifyContracts(stock)

# Request historical data (15 minute delay)
bars = ib.reqHistoricalData(
    stock,
    endDateTime='',
    durationStr='1 M',  # Requesting data 
    barSizeSetting='1 min',
    whatToShow='MIDPOINT',
    useRTH=True
)
data = util.df(bars)


from itertools import product

# Run SMA Crossover Strategy with different hyperparameters
for fast_period, slow_period in product(sma_params['fast_period'], sma_params['slow_period']):
    print(f"\nRunning SMA Crossover Strategy with fast_period={fast_period}, slow_period={slow_period}")
    sma_strategy = SmaCrossoverStrategy(stock, data, ib, {'fast_period': fast_period, 'slow_period': slow_period})
    sma_results = sma_strategy.backtest()
    sma_final_portfolio_value = sma_strategy.final_portfolio_value  # Get the final portfolio value after backtest
    sma_strategy.plot_trades()
    print(sma_strategy.trade_statistics())

# Run Bollinger Bands Strategy with different hyperparameters
for period, std_dev in product(bb_params['period'], bb_params['std_dev']):
    print(f"\nRunning Bollinger Bands Strategy with period={period}, std_dev={std_dev}")
    bb_strategy = BollingerBandsStrategy(stock, data, ib, {'period': period, 'std_dev': std_dev})
    bb_results = bb_strategy.backtest()
    bb_final_portfolio_value = bb_strategy.final_portfolio_value  # Get the final portfolio value after backtest
    bb_strategy.plot_trades()
    print(bb_strategy.trade_statistics())

ib.disconnect()
