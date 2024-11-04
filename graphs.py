from backtester import Backtester
if __name__ == "__main__":
    sma_params = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 40],
        'take_profit_pct': [0.05],
        'stop_loss_pct': [0.03]
    }

    bb_params = {
        'period': [15, 20, 25],
        'std_dev': [1, 1.5, 2, 2.5],
        'take_profit_pct': [0.05],
        'stop_loss_pct': [0.03]
    }

    sbb_params = {
        'period': [15, 20, 25],
        'std_dev': [1, 1.5, 2, 2.5],
        'rsi_window': [5, 14],
        'take_profit_pct': [0.05],
        'stop_loss_pct': [0.03]
    }

    drv_params = {
        'period': [i for i in range(5, 31,2)],
        'std_dev': [.5, 1, 1.5, 2],
        'take_profit_pct': [0.05],
        'stop_loss_pct': [0.03]
    }


    backtester = Backtester('CPNG', 'SMART', 'USD')
    # backtester.run_sma_strategy(sma_params)
    # backtester.run_bb_strategy(bb_params)
    # backtester.run_sbb_strategy(sbb_params)
    backtester.run_drv_strategy(drv_params)
    backtester.disconnect()
