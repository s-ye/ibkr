import os
import random
import pandas as pd
from datetime import timedelta
from itertools import product
from ib_insync import IB, Stock, util
from strategies import SmaCrossoverStrategy, BollingerBandsStrategy, DipRecoverVolumeStrategy, SidewaysBollingerBandsStrategy

class Backtester:
    def __init__(self, stock_symbol, exchange, currency, client_id=1):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=client_id)
        self.stock = Stock(stock_symbol, exchange, currency)
        self.ib.qualifyContracts(self.stock)
        self.full_data = self._get_full_historical_data(stock_symbol)
        self.data = self._get_historical_data(stock_symbol)

    def _get_full_historical_data(self, stock_symbol):
        cache_file = f"cache/{stock_symbol}_2year_15min_data.csv"
        if os.path.exists(cache_file):
            print("Loading full historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching 2 Y of historical 15-minute data from IBKR...")
        bars = self.ib.reqHistoricalData(
            self.stock,
            endDateTime='',
            durationStr='2 Y',
            barSizeSetting='15 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        df = util.df(bars)
        df.set_index('date', inplace=True)
        df.to_csv(cache_file)
        return df

    def run_sampled_backtests(self, num_samples=10, duration_days=30, sma_params=None, bb_params=None,sbb_params = None, drv_params=None):
        results = []
        
        for _ in range(num_samples):
            sample_data = self._get_random_sample(duration_days)
            
            # Run SMA Crossover Strategy with different parameter combinations
            if sma_params:
                for fast_period, slow_period, take_profit_pct, stop_loss_pct in product(
                        sma_params['fast_period'],
                        sma_params['slow_period'],
                        sma_params['take_profit_pct'],
                        sma_params['stop_loss_pct']
                    ):
                    sma_strategy = SmaCrossoverStrategy(
                        self.stock, sample_data, self.ib,
                        params={'fast_period': fast_period, 'slow_period': slow_period},
                        initial_capital=1_000_000,
                        profit_target_pct=take_profit_pct,
                        trailing_stop_pct=stop_loss_pct
                    )
                    sma_strategy.backtest()
                    stats = sma_strategy.trade_statistics()
                    results.append({
                        'strategy': 'SMA_Crossover',
                        'params': {'fast_period': fast_period, 'slow_period': slow_period, 
                                   'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct},
                        'start_date': sample_data.index[0],
                        'end_date': sample_data.index[-1],
                        'final_portfolio_value': stats['final_portfolio_value'],
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'losing_trades': stats['losing_trades'],
                        'average_return': stats['average_return'],
                    })

            # Run Bollinger Bands Strategy with different parameter combinations
            if bb_params:
                for period, std_dev, take_profit_pct, stop_loss_pct in product(
                        bb_params['period'],
                        bb_params['std_dev'],
                        bb_params['take_profit_pct'],
                        bb_params['stop_loss_pct']
                    ):
                    bb_strategy = BollingerBandsStrategy(
                        self.stock, sample_data, self.ib,
                        params={'period': period, 'std_dev': std_dev},
                        initial_capital=1_000_000,
                        profit_target_pct=take_profit_pct,
                        trailing_stop_pct=stop_loss_pct
                    )
                    bb_strategy.backtest()
                    stats = bb_strategy.trade_statistics()
                    results.append({
                        'strategy': 'Bollinger_Bands',
                        'params': {'period': period, 'std_dev': std_dev, 
                                   'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct},
                        'start_date': sample_data.index[0],
                        'end_date': sample_data.index[-1],
                        'final_portfolio_value': stats['final_portfolio_value'],
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'losing_trades': stats['losing_trades'],
                        'average_return': stats['average_return'],
                    })
            if sbb_params:
                for period, std_dev, rsi_window, take_profit_pct, stop_loss_pct in product(
                        sbb_params['period'],
                        sbb_params['std_dev'],
                        sbb_params['rsi_window'],
                        sbb_params['take_profit_pct'],
                        sbb_params['stop_loss_pct']
                    ):
                    sbb_strategy = SidewaysBollingerBandsStrategy(
                        self.stock, sample_data, self.ib,
                        params={'period': period, 'std_dev': std_dev, 'rsi_window': rsi_window},
                        initial_capital=1_000_000,
                        profit_target_pct=take_profit_pct,
                        trailing_stop_pct=stop_loss_pct
                    )
                    sbb_strategy.backtest()
                    stats = sbb_strategy.trade_statistics()
                    results.append({
                        'strategy': 'Sideways_Bollinger_Bands',
                        'params': {'period': period, 'std_dev': std_dev, 'rsi_window': rsi_window,
                                   'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct},
                        'start_date': sample_data.index[0],
                        'end_date': sample_data.index[-1],
                        'final_portfolio_value': stats['final_portfolio_value'],
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'losing_trades': stats['losing_trades'],
                        'average_return': stats['average_return'],
                    })

            # Run Morning Dip Afternoon Recovery Strategy with different parameter combinations
            if drv_params:
                for period, std_dev, take_profit_pct, stop_loss_pct in product(
                        drv_params['period'],
                        drv_params['std_dev'],
                        drv_params['take_profit_pct'],
                        drv_params['stop_loss_pct']
                    ):
                    drv_strategy = DipRecoverVolumeStrategy(
                        self.stock, sample_data, self.ib,
                        params={'period': period, 'std_dev': std_dev},
                        initial_capital=1_000_000,
                        profit_target_pct=take_profit_pct,
                        trailing_stop_pct=stop_loss_pct
                    )
                    drv_strategy.backtest()
                    stats = drv_strategy.trade_statistics()
                    results.append({
                        'strategy': 'Dip_Recovery_Volume',
                        'params': {'period': period, 'std_dev': std_dev, 
                                   'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct},
                        'start_date': sample_data.index[0],
                        'end_date': sample_data.index[-1],
                        'final_portfolio_value': stats['final_portfolio_value'],
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'losing_trades': stats['losing_trades'],
                        'average_return': stats['average_return'],
                    })
                

        # Convert results to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Perform grouping operation
        results_df['params'] = results_df['params'].apply(lambda x: str(x))
        average_results = results_df.groupby(['strategy', 'params'])['final_portfolio_value'].mean()
        print("Average Results:\n", average_results)
                # Sort the results within each strategy by average final portfolio value in descending order
        average_results = average_results.sort_values(ascending=False)

        # Optionally, reset the index to make the grouped result easier to view as a DataFrame
        average_results = average_results.reset_index()

        # Display the results
        print("Average performance for each strategy and parameter configuration:")
        print(average_results)
        
        return results_df, average_results




    def _get_random_sample(self, duration_days):
        """Get a random 1-month (or custom duration) sample from the 3-year dataset."""
        total_bars = len(self.full_data)
        bars_per_day = int(timedelta(days=1) / timedelta(minutes=15))  # 96 bars for 15-min intervals
        sample_size = bars_per_day * duration_days
        
        # Select a random start index that allows for a full sample within bounds
        start_idx = random.randint(0, total_bars - sample_size)
        sample_data = self.full_data.iloc[start_idx:start_idx + sample_size].copy()
        
        return sample_data

    def disconnect(self):
        self.ib.disconnect()

    def _get_historical_data(self, stock_symbol):

        cache_file = f"cache/{stock_symbol}_1month_15min_data.csv"
        if os.path.exists(cache_file):
            print("Loading historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching 1 M of historical 15-minute data from IBKR...")
        bars = self.ib.reqHistoricalData(
            self.stock,
            endDateTime='',
            durationStr='1 M',
            barSizeSetting='15 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        df = util.df(bars)
        df.to_csv(cache_file)
        # index by date
        df.set_index('date', inplace=True)
        return df

    def run_sma_strategy(self, sma_params):
        for fast_period, slow_period, take_profit_pct, stop_loss_pct in product(
                sma_params['fast_period'],
                sma_params['slow_period'],
                sma_params['take_profit_pct'],
                sma_params['stop_loss_pct']
            ):
            print(f"\nRunning SMA Crossover Strategy with fast_period={fast_period}, slow_period={slow_period}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            sma_strategy = SmaCrossoverStrategy(
                self.stock, self.data, self.ib,
                params={'fast_period': fast_period, 'slow_period': slow_period},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            sma_strategy.backtest()
            stats = sma_strategy.trade_statistics()
            print(stats)
            sma_strategy.plot_trades()
            

    def run_bb_strategy(self, bb_params):
        for period, std_dev, take_profit_pct, stop_loss_pct in product(
                bb_params['period'],
                bb_params['std_dev'],
                bb_params['take_profit_pct'],
                bb_params['stop_loss_pct']
            ):
            print(f"\nRunning Bollinger Bands Strategy with period={period}, std_dev={std_dev}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            bb_strategy = BollingerBandsStrategy(
                self.stock, self.data, self.ib,
                params={'period': period, 'std_dev': std_dev},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            bb_strategy.backtest()
            stats = bb_strategy.trade_statistics()
            print(stats)
            bb_strategy.plot_trades()

    def run_sbb_strategy(self, sbb_params):
        for period, std_dev, rsi_window, take_profit_pct, stop_loss_pct in product(
                sbb_params['period'],
                sbb_params['std_dev'],
                sbb_params['rsi_window'],
                sbb_params['take_profit_pct'],
                sbb_params['stop_loss_pct']
            ):
            print(f"\nRunning Sideways Bollinger Bands Strategy with period={period}, std_dev={std_dev}, rsi_window={rsi_window}, "
                  f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")
            sbb_strategy = SidewaysBollingerBandsStrategy(
                self.stock, self.data, self.ib,
                params={'period': period, 'std_dev': std_dev, 'rsi_window': rsi_window},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )
            sbb_strategy.backtest()
            stats = sbb_strategy.trade_statistics()
            print(stats)


            sbb_strategy.plot_trades()




    def run_drv_strategy(self, drv_params):
        for period, std_dev, take_profit_pct, stop_loss_pct in product(
            drv_params['period'],
            drv_params['std_dev'],
            drv_params['take_profit_pct'],
            drv_params['stop_loss_pct']
        ):
            print(f"Running Dip Recovery Volume Strategy with period={period}, std_dev={std_dev}, "
                f"take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")


            # Instantiate a new strategy instance
            drv_strategy = DipRecoverVolumeStrategy(
                self.stock, self.data, self.ib,
                params={'period': period, 'std_dev': std_dev},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )

            # Run the backtest and gather results
            drv_strategy.backtest()
            stats = drv_strategy.trade_statistics()

            # Append results with checks
            print(stats)

            drv_strategy.plot_trades()
