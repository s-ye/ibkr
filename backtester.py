# Standard library imports
import os
import random
from datetime import timedelta
from itertools import product

from strategies import GBMStrategy
import pandas as pd
from ib_insync import IB, Stock, util

# Local imports
from strategies import *

class Backtester:
    def __init__(self, stock_symbol, exchange, currency):
        self.ib = None
        self.stock = Stock(stock_symbol, exchange, currency)
        # Ensure cache directory exists
        if not os.path.exists('cache'):
            os.makedirs('cache')
        self.data_retrieve = Data_Retrieve()
        self.two_yr_15min_data = self.data_retrieve._get_2yr_15min_data(self.stock)
        self.one_mo_15min_data = self.data_retrieve._get_1mo_15min_data(self.stock)

        self.max_1d_data = self.data_retrieve._get_max_1d_data(self.stock)
        self.five_yr_1d_data = self.max_1d_data[-1260:].copy()
        self.one_yr_1d_data = self.five_yr_1d_data[-252:].copy()
        self.six_mo_1d_data = self.five_yr_1d_data[-126:].copy()
        self.three_mo_1d_data = self.five_yr_1d_data[-63:].copy()

    def _get_full_historical_data(self, stock_symbol, exchange, currency):
        cache_file = f"cache/{stock_symbol}_2year_15min_data.csv"
        if os.path.exists(cache_file):
            print("Loading full historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching 2 Y of historical 15-minute data from IBKR...")
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.ib.qualifyContracts(self.stock)
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
        # Ensure cache directory exists
        if not os.path.exists('cache'):
            os.makedirs('cache')
        df.to_csv(cache_file)
        self.ib.disconnect()
        return df
        

    def run_sampled_backtests(self, num_samples=100, duration_days=30,gbm_params=None):
        results = []
        for _ in range(num_samples):
            sample_data = self._get_random_sample(duration_days)
            if gbm_params:
                for threshold, time_periods, num_simulations, take_profit_pct, stop_loss_pct in product(
                        gbm_params['threshold'],
                        gbm_params['time_periods'],
                        gbm_params['num_simulations'],
                        gbm_params['take_profit_pct'],
                        gbm_params['stop_loss_pct']
                    ):
                    gbm_strategy = GBMStrategy(
                        self.stock, sample_data, self.ib,
                        params={'threshold': threshold, 'time_periods': time_periods, 'num_simulations': num_simulations},
                        initial_capital=1_000_000,
                        profit_target_pct=take_profit_pct,
                        trailing_stop_pct=stop_loss_pct
                    )
                    gbm_strategy.backtest()
                    stats = gbm_strategy.trade_statistics()
                    gbm_strategy.plot_trades()
                    
                    results.append({
                        'strategy': 'Geometric_Brownian_Motion',
                        'params': {'threshold': threshold, 'time_periods': time_periods, 'num_simulations': num_simulations,
                                   'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct},
                        'start_date': sample_data.index[0],
                        'end_date': sample_data.index[-1],
                        'final_portfolio_value': stats['final_portfolio_value'],
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'losing_trades': stats['losing_trades'],
                        'average_return': stats['average_return'],
                        'sharpe_ratio': stats['sharpe_ratio'],
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
        """Get a random 1-month (or custom duration) sample from the 2-year dataset."""
        total_bars = len(self.two_yr_15min_data)
        # bars_per_day based on 930am-4pm trading hours
        bars_per_day = 26
        sample_size = bars_per_day * duration_days
        
        # Select a random start index that allows for a full sample within bounds
        if sample_size > total_bars:
            raise ValueError("Sample size is larger than the total available data.")
        start_idx = random.randint(0, total_bars - sample_size)
        sample_data = self.two_yr_15min_data.iloc[start_idx:start_idx + sample_size].copy()
        
        return sample_data


    def run_gbm_strategy(self, gbm_params):
        for threshold, time_periods, num_simulations, take_profit_pct, stop_loss_pct in product(
            gbm_params['threshold'],
            gbm_params['time_periods'],
            gbm_params['num_simulations'],
            gbm_params['take_profit_pct'],
            gbm_params['stop_loss_pct']
        ):
            print(f"Running Geometric Brownian Motion Strategy with threshold={threshold}, time_periods={time_periods}, "
                f"num_simulations={num_simulations}, take_profit_pct={take_profit_pct}, stop_loss_pct={stop_loss_pct}")


            # Instantiate a new strategy instance
            gbm_strategy = GBMStrategy(
                self.stock, self.one_mo_15min_data, self.ib,
                params={'threshold': threshold, 'time_periods': time_periods, 'num_simulations': num_simulations},
                initial_capital=1_000_000,
                profit_target_pct=take_profit_pct,
                trailing_stop_pct=stop_loss_pct
            )

            # Run the backtest and gather results
            gbm_strategy.backtest()
            stats = gbm_strategy.trade_statistics()
            gbm_strategy.plot_trades()

    def forecast_15_mins(self,gbm_params):
        gbm_strategy = GBMStrategy(
            self.stock, self.one_mo_15min_data,
            self.ib,
            params={'threshold': gbm_params['threshold'], 'time_periods': gbm_params['time_periods'], 'num_simulations': gbm_params['num_simulations']},
            initial_capital=1_000_000,
            profit_target_pct=0.02,
            trailing_stop_pct=0.02
        )
        simulations, forecast_dates = gbm_strategy.forecast()
        # plot a random simulation

        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.plot(forecast_dates, simulations[i], alpha=0.5)
        plt.title(f"GBM Forecast for {self.stock.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()




    def forecast_1_day_1yr(self,gbm_params):
        gbmd_strategy = GBMStrategy(
            self.stock, self.one_yr_1d_data, self.ib,
            params={'threshold': gbm_params['threshold'], 'time_periods': gbm_params['time_periods'], 'num_simulations': gbm_params['num_simulations']},
            initial_capital=1_000_000,
            profit_target_pct=0.02,
            trailing_stop_pct=0.02
        )
        simulations, forecast_dates = gbmd_strategy.forecast()
        # plot a random simulation

        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.plot(forecast_dates, simulations[i], alpha=0.5)
        plt.title(f"GBM Forecast for {self.stock.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

        return simulations, forecast_dates

    def forecast_1_day_6m(self,gbm_params):
        gbmd_strategy = GBMStrategy(
            self.stock, self.six_mo_1d_data, self.ib,
            params={'threshold': gbm_params['threshold'], 'time_periods': gbm_params['time_periods'], 'num_simulations': gbm_params['num_simulations']},
            initial_capital=1_000_000,
            profit_target_pct=0.02,
            trailing_stop_pct=0.02
        )
        simulations, forecast_dates = gbmd_strategy.forecast()
        # plot a random simulation

        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.plot(forecast_dates, simulations[i], alpha=0.5)
        plt.title(f"GBM Forecast for {self.stock.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

        return simulations, forecast_dates

    def forecast_1_day_3m(self,gbm_params):
        gbmd_strategy = GBMStrategy(
            self.stock, self.three_mo_1d_data, self.ib,
            params={'threshold': gbm_params['threshold'], 'time_periods': gbm_params['time_periods'], 'num_simulations': gbm_params['num_simulations']},
            initial_capital=1_000_000,
            profit_target_pct=0.02,
            trailing_stop_pct=0.02
        )
        simulations, forecast_dates = gbmd_strategy.forecast()
        # plot a random simulation

        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.plot(forecast_dates, simulations[i], alpha=0.5)
        plt.title(f"GBM Forecast for {self.stock.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

        return simulations, forecast_dates
    
    def forecast_data(self,gbm_params,data):
        gbm_strategy = GBMStrategy(
            self.stock, data, self.ib,
            params={'threshold': gbm_params['threshold'], 'time_periods': gbm_params['time_periods'], 'num_simulations': gbm_params['num_simulations']},
            initial_capital=1_000_000,
            profit_target_pct=0.02,
            trailing_stop_pct=0.02
        )
        simulations, associated_mu, associated_sigma, forecast_dates = gbm_strategy.forecast()  
        # plot a random simulation

        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.plot(forecast_dates, simulations[i], alpha=0.5)
            # label each simulation with its associated mu and sigma
            plt.text(forecast_dates[-1], simulations[i][-1], f"μ={associated_mu[i]:.2f}, σ={associated_sigma[i]:.2f}")
        plt.title(f"GBM Forecast for {self.stock.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

        return simulations, associated_mu, associated_sigma, forecast_dates


class Data_Retrieve:
    def __init__(self):
        self.ib = IB()


    def _get_2yr_15min_data(self,stock):
        cache_file = f"cache/{stock.symbol}_2year_15min_data.csv"
        if os.path.exists(cache_file):
            print("Loading historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching 2 Y of historical 15-minute data from IBKR...")
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.stock = stock
        self.ib.qualifyContracts(self.stock)
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
        # Ensure cache directory exists
        if not os.path.exists('cache'):
            os.makedirs('cache')
        df.to_csv(cache_file)
        self.ib.disconnect()
        return df
    
    def _get_1mo_15min_data(self, stock):

        cache_file = f"cache/{stock.symbol}_1month_15min_data.csv"
        if os.path.exists(cache_file):
            print("Loading historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching 1 M of historical 15-minute data from IBKR...")
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.stock = stock
        self.ib.qualifyContracts(self.stock)
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
        self.ib.disconnect()
        return df

    
    def _get_max_1d_data(self, stock):
        cache_file = f"cache/{stock.symbol}_max_1day_data.csv"
        if os.path.exists(cache_file):
            print("Loading historical data from cache...")
            return pd.read_csv(cache_file, index_col='date', parse_dates=True)
        
        print("Fetching max historical daily data from IBKR...")
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.stock = stock
        self.ib.qualifyContracts(self.stock)
        bars = self.ib.reqHistoricalData(
            self.stock,
            endDateTime='',
            # Max historical data
            durationStr='15 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        df = util.df(bars)
        df.to_csv(cache_file)
        # index by date
        df.set_index('date', inplace=True)
        self.ib.disconnect()
        return df

