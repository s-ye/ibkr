from backtester import Backtester

stock = 'EPD'

if __name__ == "__main__":
    gbm_params = {
        'threshold': .2,
        'time_periods': 365,
        'num_simulations': 150
        # 'take_profit_pct': [0.02],
        # 'stop_loss_pct': [0.02]
    }

    backtester = Backtester(stock, 'SMART', 'USD')
    backtester.forecast_1_day(gbm_params)

