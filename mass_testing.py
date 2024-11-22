from backtester import Backtester

stock = 'WMT'

    # Example usage
if __name__ == "__main__":
    # .2, 30, 150, .02, .02
    gbm_params = {
        'threshold': [.2,.3,.4],
        'time_periods': [30],
        'num_simulations': [150],
        'take_profit_pct': [0.01,0.02],
        'stop_loss_pct': [0.02]
    }

    backtester = Backtester(stock, 'SMART', 'USD')


    # Run backtests with sampled periods and parameter grids
    results_df, average_results = backtester.run_sampled_backtests(
        num_samples=15, duration_days=30, gbm_params=gbm_params
    )


    # Step 2: Rename columns for clarity
    average_results.rename(columns={
        'final_portfolio_value': 'Final Portfolio Value',
        'period': 'Period',
        'std_dev': 'Standard Deviation',
        'take_profit_pct': 'Take Profit %',
        'stop_loss_pct': 'Stop Loss %'
    }, inplace=True)

    # Step 4: Format float columns to two decimal places for readability
    average_results['Final Portfolio Value'] = average_results['Final Portfolio Value'].round(2)


    # Step 5: Save the formatted DataFrame to CSV
    output_file = f'results/{stock}_average_results.csv'
    average_results.to_csv(output_file, index=False)

    print(f"Formatted average results saved to {output_file}")


    # Save results for further analysis
    results_df.to_csv("results/results.csv")

    # process the results
    # for each start_date, I want to see the strategy and parameters that performed the best
    sorted_results = results_df.sort_values(by=['start_date', 'final_portfolio_value'], ascending=[True, False])

    # Group by 'start_date' and get the top 3 strategies for each date
    top_3_per_start_date = sorted_results.groupby('start_date').head(3)

    top_3_per_start_date = sorted_results.groupby('start_date').head(3).reset_index(drop=True)

    # Step 2: Add a rank column for readability
    top_3_per_start_date['Rank'] = top_3_per_start_date.groupby('start_date').cumcount() + 1

    # Step 3: Rename columns to be more descriptive
    top_3_per_start_date.rename(columns={
        'start_date': 'Start Date',
        'strategy': 'Strategy Name',
        'params': 'Parameters',
        'final_portfolio_value': 'Final Portfolio Value'
    }, inplace=True)

    # Step 4: Reorder columns for readability
    top_3_per_start_date = top_3_per_start_date[['Start Date', 'Rank', 'Strategy Name', 'Parameters', 'Final Portfolio Value']]

    # Step 5: Export to CSV with specific formatting options
    output_file = f'results/{stock}_top_strategies_by_start_date.csv'
    top_3_per_start_date.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Top 3 strategies per start date saved to {output_file}")


    backtester.disconnect()