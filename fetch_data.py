from backtester import Backtester

stocks = ['EAT','SG','WMT', 'QQQ']
for stock in stocks:
    backtester = Backtester(stock, 'SMART', 'USD')