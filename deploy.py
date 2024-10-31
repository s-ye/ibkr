from ib_insync import IB, Stock, util
from strategies import BollingerBandsStrategy
import time
import pandas as pd

# Connect to IBKR paper trading account
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define stock and parameters
stock = Stock('NVO', 'SMART', 'USD')
ib.qualifyContracts(stock)

# Fetch historical data and initialize the strategy
bars = ib.reqHistoricalData(stock, endDateTime='', durationStr='2 D', barSizeSetting='15 mins', whatToShow='MIDPOINT', useRTH=True)
data = util.df(bars)

# Initialize the Bollinger Bands strategy with specified parameters
bollinger_params = {'period': 20, 'std_dev': 1.5}
strategy = BollingerBandsStrategy(stock, data, ib, params=bollinger_params, profit_target_pct=0.1, trailing_stop_pct=0.03)

# Function to update data and run paper trading
def update_data_and_trade():
    print("Fetching latest data...")
    latest_bars = ib.reqHistoricalData(
        stock, 
        endDateTime='', 
        durationStr='1 D', 
        barSizeSetting='15 mins', 
        whatToShow='MIDPOINT', 
        useRTH=True
    )
    latest_data = util.df(latest_bars)
    strategy.data = pd.concat([strategy.data, latest_data]).drop_duplicates(subset=['date'])
    
    print("Generating signals...")
    strategy.data_with_signals = strategy.generate_signals()
    
    print("Running paper trading...")
    strategy.run_paper_trading()  # Run paper trading based on the latest signals
    print("Update complete.")

# Run in a loop to check for signals every 15 minutes
try:
    while True:
        update_data_and_trade()
        time.sleep(900) # Sleep for 15 minutes

except KeyboardInterrupt:
    ib.disconnect()
    print("Disconnected from IBKR.")