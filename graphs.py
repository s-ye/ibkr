from backtester import Backtester
if __name__ == "__main__":
    gbm_params = {
        'threshold': [1.5],
        'time_periods': [100],
        'num_simulations': [150],
        'take_profit_pct': [0.005],
        'stop_loss_pct': [0.01]
    }

    
    long_tickers = ['CEG', 'LH', 'BA', 'CARR', 'DOW', 'PH', 'EMR', 'JBL', 'SWK', 'URI', 'BSX', 'DLTR', 'ORCL', 'HUBB', 'LYB', 'XYL', 'HON',
                    'DD', 'ROP', 'UNH', 'IBM', 'GRMN', 'CMI', 'BKR', 'GLW', 'SYK', 'FTV', 'ETN', 'CHD', 'OTIS', 'PCAR', 'DGX', 'AME',
                    'DRI', 'APH', 'AOS', 'HUM', 'CLX', 'ORLY', 'CTAS', 'ECL', 'TER', 'TMUS', 'MAS', 'TDG', 'JNPR', 'NSC', 'FAST', 'PAYX',
                    'ROK', 'ITW', 'CSCO', 'CPRT', 'TMO', 'OKE', 'EXC', 'EMN', 'PWR', 'NEM', 'DOV', 'VTR', 'TXT', 'TXN', 'PG', 'AVY', 'DTE',
                    'MGM', 'BR', 'GD', 'ADP', 'PPL', 'NI', 'MLM', 'IDXX', 'HCA', 'SHW', 'HWM', 'ZTS', 'RCL', 'GWW', 'CDW', 'CAH', 'HPE', 'HD',
                    'HSY', 'RTX', 'UNP', 'MCK', 'AES', 'FICO', 'INTC', 'JCI', 'ATO', 'HAS', 'LOW', 'ALLE', 'WELL', 'ISRG', 'VRSN', 'TRGP', 'LMT']
    short_tickers = ['ETSY', 'DXCM', 'ILMN', 'PAYC', 'VFC', 'ABNB', 'APA', 'UPS', 'EPAM', 'CHTR', 'MOS', 'EXPE', 'MPC', 'PANW', 'VLO', 'COR', 'BXP', 'MRO', 'HAL', 'MRNA']

    # scramble the order
    import random
    random.shuffle(long_tickers)
    random.shuffle(short_tickers)

    for ticker in long_tickers:
        backtester = Backtester(ticker, 'SMART', 'USD')
        backtester.run_gbm_strategy(gbm_params)


    for ticker in short_tickers:
        backtester = Backtester(ticker, 'SMART', 'USD')
        backtester.run_gbm_strategy(gbm_params)

