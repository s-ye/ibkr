from backtester import Backtester

stock = 'WMT'

if __name__ == "__main__":
    gbm_params = {
        'threshold': .2,
        'time_periods': 365,
        'num_simulations': 150
    }
    backtester = Backtester(stock, 'SMART', 'USD')
    backtester.forecast_1_day(gbm_params)
    backtester.forecast_15_mins(gbm_params)

