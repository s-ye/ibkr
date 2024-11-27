from backtester import Backtester

stock = 'TGT'

if __name__ == "__main__":
    gbm_params = {
        'threshold': 2,
        'time_periods': 90,
        'num_simulations': 10000,
    }
    backtester = Backtester(stock, 'SMART', 'USD')
    backtester.forecast_1_day_1yr(gbm_params)
    backtester.forecast_1_day_3m(gbm_params)
    backtester.forecast_15_mins(gbm_params)

