from ib_insync import IB, Stock, Ticker
from datetime import datetime
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            connect_ib()  # Retry connection

def request_real_time_data(stock_symbol, exchange='SMART', currency='USD', on_price_update=None):
    """Request real-time market data for a stock."""
    # Define the stock
    stock = Stock(stock_symbol, exchange, currency)
    ib.qualifyContracts(stock)
    
    # Request real-time market data
    ticker = ib.reqMktData(stock, '', False, False)
    logger.info(f"Requested real-time data for {stock_symbol}")

    # Attach a callback if provided
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
    """Callback function to process real-time price updates."""
    current_price = ticker.last
    if current_price is not None:
        current_time = datetime.now()
        logger.info(f"Updating strategy with price: {current_price} at {current_time}")
        # Here you could call your strategy's update method, e.g.,
        # strategy.update_with_price(current_price, current_time)
        # strategy.run_paper_trading()  # or strategy.run_live_trading()

# Usage example (you can import these functions in deploy.py or main.py):
if __name__ == '__main__':
    connect_ib()
    ticker = request_real_time_data(stock_symbol='AAPL', on_price_update=on_price_update)
    start_event_loop()