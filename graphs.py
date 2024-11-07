from backtester import Backtester
if __name__ == "__main__":
    gbm_params = {
        'threshold': [0.1,0.2,.5,1,2],
        'time_periods': [45],
        'num_simulations': [150],
        'take_profit_pct': [0.05],
        'stop_loss_pct': [0.03]
    }
    backtester = Backtester('MRNA', 'SMART', 'USD')
    backtester.run_gbm_strategy(gbm_params)
    backtester.disconnect()
