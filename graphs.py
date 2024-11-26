from backtester import Backtester
if __name__ == "__main__":
    gbm_params = {
        'threshold': [1.5],
        'time_periods': [100],
        'num_simulations': [150],
        'take_profit_pct': [0.005],
        'stop_loss_pct': [0.01]
    }

    
    stock = 'MRNA'
    backtester = Backtester(stock, 'SMART', 'USD')
    backtester.run_gbm_strategy(gbm_params)
    
    
