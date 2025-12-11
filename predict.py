import pandas as pd
import os
import glob
import pickle
from autogluon.timeseries import TimeSeriesDataFrame

import numpy as np


def run_forecasting(tsv_folder_path, forecast_results_output_path):
    csv_files = glob.glob(os.path.join(tsv_folder_path, "*.csv"))
    latest_file = max(csv_files, key=os.path.getmtime)

    df = pd.read_csv(latest_file, sep='\t', parse_dates=['date'])
    df.columns = ['timestamp', 'target']
    df['item_id'] = 'series_1'

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')

    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    forecast = best_model.predict(ts_df, prediction_length=90)

    # Convert to Pandas DataFrame
    forecast_df = forecast.to_data_frame().reset_index()

    # Print and save
    print("\n--Forecast for Next 90 Days --")
    print(forecast_df)

    # Save forecast to CSV in the current working directory
    forecast_results_output_file = os.path.join(forecast_results_output_path, "forecast_output.csv")
    forecast_df.to_csv(forecast_results_output_file, index=False)

    print(f"\n Forecast written to: {forecast_results_output_path}")
    postprocess_forecast_data(forecast_results_output_file, forecast_results_output_path)



def postprocess_forecast_data(file_path, forecast_results_output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Keeping only required specified quantiles
    df_filtered = df[['timestamp', '0.3', '0.5', '0.9']].copy()
    df_filtered.rename(columns={'0.3': 'p_30', '0.5': 'forecast(p_50)', '0.9': 'p_90'}, inplace=True)

    # Round off numbers to the nearest higher integer for specified columns
    columns_to_round = ['p_30', 'forecast(p_50)', 'p_90']
    for col in columns_to_round:
        df_filtered[col] = np.ceil(df_filtered[col])

    df_filtered.to_csv(os.path.join(forecast_results_output_path, 'postprocessed_forecast_output.csv'), index=False)