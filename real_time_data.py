from ib_insync import IB, Stock, Ticker, util
from datetime import datetime
from strategies import BollingerBandsStrategy
import logging
import time
import os


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Connect to IBKR
ib = IB()

def connect_ib():
    """Connect to IBKR and handle connection errors."""
    if not ib.isConnected():
        try:
            ib.connect('127.0.0.1', 7497, clientId=1)
            logger.info("Connected to IBKR.")
        except Exception as e:
            logger.error(f"Connection error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            connect_ib()

def get_initial_capital_and_position(stock_symbol):
    """Fetch initial capital and current position for a specified stock from IBKR."""
    # Retrieve account cash balance as initial capital
    account_value = ib.accountValues()
    cash_balance = next((float(item.value) for item in account_value if item.tag == 'AvailableFunds'), 0.0)

    # Retrieve current position in the specified stock
    positions = ib.positions()
    aapl_position = next((pos.position for pos in positions if pos.contract.symbol == stock_symbol), 0)

    return cash_balance, aapl_position

def request_real_time_data(stock_symbol, exchange='SMART', currency='USD', on_price_update=None):
    """Request real-time market data for a stock."""
    stock = Stock(stock_symbol, exchange, currency)
    ib.qualifyContracts(stock)
    ticker = ib.reqMktData(stock, '', False, False)
    logger.info(f"Requested real-time data for {stock_symbol}")

    if on_price_update:
        ticker.updateEvent += on_price_update
    return ticker

def start_event_loop():
    """Start the IBKR event loop for real-time updates."""
    try:
        ib.run()
    except KeyboardInterrupt:
        ib.disconnect()
        logger.info("Disconnected from IBKR.")

def on_price_update(ticker: Ticker):
    """Callback function to process real-time price updates and log buys/sells."""
    current_price = ticker.last
    if current_price is not None:
        current_time = datetime.now()
        logger.info(f"Updating strategy with price: {current_price} at {current_time}")
        
        # Store the number of trades before running the update
        initial_trade_count = len(strategy.trades)
        
        # Update the strategy with the new price point and attempt to make trades
        strategy.update_with_price(current_price, current_time)
        strategy.run_paper_trading()

        # Check if a new trade was made by comparing trade count
        if len(strategy.trades) > initial_trade_count:
            last_trade = strategy.trades[-1]
            trade_type = "Buy" if last_trade['shares'] > 0 else "Sell"
            strategy.logger.info(
                f"{trade_type} executed - Entry Date: {last_trade['entry_date']}, "
                f"Exit Date: {last_trade['exit_date']}, Entry Price: {last_trade['entry_price']}, "
                f"Exit Price: {last_trade['exit_price']}, Shares: {last_trade['shares']}, "
                f"Profit: {last_trade['profit']}, Duration: {last_trade['duration']} mins"
            )
        else:
            logger.debug("No new trade executed.")
        # Add a delay between price updates
        time.sleep(5)


# Main script to initialize and run the strategy
if __name__ == '__main__':
    connect_ib()
    
    # Initialize the stock and fetch historical data
    stock_symbol = 'AAPL'
    stock = Stock(stock_symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    
    bars = ib.reqHistoricalData(
        stock, endDateTime='', durationStr='2 D', barSizeSetting='15 mins',
        whatToShow='MIDPOINT', useRTH=True
    )
    data = util.df(bars)
    
    # Initialize the strategy with parameters
    strategy_params = {
        'period': 20,
        'std_dev': 1.5,
        'profit_target_pct': 0.05,
        'trailing_stop_pct': 0.03
    }
    # calculate initial capital and position
    initial_capital, position = get_initial_capital_and_position(stock_symbol)
    strategy = BollingerBandsStrategy(stock, data, ib, params=strategy_params, initial_capital=initial_capital)
    strategy.current_position = position  # Set the current AAPL position



    # Request real-time data and start event loop
    ticker = request_real_time_data(stock_symbol=stock_symbol, on_price_update=on_price_update)
    start_event_loop()