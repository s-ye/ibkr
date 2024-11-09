import pandas as pd
from backtester import Backtester
from gbm import GBMGridSearch

# Assume you have already loaded MRNA 15-minute bar data into a DataFrame
# For demonstration, let's say the DataFrame is named `mrna_data` and has columns 'timestamp' and 'close'
backtester = Backtester('MRNA', 'SMART', 'USD')
mrna_data = backtester.full_data 
backtester.disconnect()

# Filter data for the training and testing period
# Assuming 'timestamp' is datetime and sorted in ascending order

# Ensure 'timestamp' column is in datetime format and set it as the index
mrna_data.index = pd.to_datetime(mrna_data.index, utc=True)  # Set UTC if needed for consistency

# Define time ranges in a compatible format (also timezone-aware if mrna_data.index is timezone-aware)
train_start_date = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=4)
train_end_date = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=1)
test_start_date = train_end_date
test_end_date = pd.Timestamp.now(tz="UTC")

# Now perform the train-test split with timezone-consistent filtering
train_data = mrna_data[(mrna_data.index >= train_start_date) & (mrna_data.index < train_end_date)]
test_data = mrna_data[(mrna_data.index >= test_start_date) & (mrna_data.index <= test_end_date)]

# Split data into training and testing sets
mrna_data.index = pd.to_datetime(mrna_data.index)


train_data = mrna_data[(mrna_data.index >= train_start_date) & (mrna_data.index < train_end_date)]
test_data = mrna_data[(mrna_data.index >= test_start_date) & (mrna_data.index <= test_end_date)]

# Define hyperparameter ranges for grid search
# mu_means = [0.001, 0.005, 0.01, .02, 0.05]
# mu_stds = [0.01, 0.02, 0.05, 0.1]
# sigma_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1,2]

mu_means = [0.001, 0.005, 0.01]
mu_stds = [0.01, 0.02,.01]
sigma_scales = [0.01, 0.05]
num_samples = [2000]
burn_in_periods = [1000]
chains = [6]

# Initialize GBMGridSearch with the training data
grid_search = GBMGridSearch(
    historical_data=train_data,
    mu_means=mu_means,
    mu_stds=mu_stds,
    sigma_scales=sigma_scales,
    num_samples=num_samples,
    burn_in_periods=burn_in_periods,
    chains=chains
)

# Run the grid search
grid_search.run_grid_search(test_data)

# Log the results
grid_search.log_results()