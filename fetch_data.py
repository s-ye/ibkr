from backtester import Backtester

stocks = ['EAT','SG']
for stock in stocks:
    backtester = Backtester(stock, 'SMART', 'USD')