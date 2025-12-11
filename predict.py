import pandas as pd
import os
import glob
import pickle
from autogluon.timeseries import TimeSeriesDataFrame


def run_forecasting(tsv_folder_path):
    csv_files = glob.glob(os.path.join(tsv_folder_path, "*.csv"))
    latest_file = max(csv_files, key=os.path.getmtime)

    df = pd.read_csv(latest_file, sep='\t', parse_dates=['date'])
    df.columns = ['timestamp', 'target']
    df['item_id'] = 'series_1'

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')

    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    forecast = best_model.predict(ts_df, prediction_length=90)
    print("\n===== Forecast for Next 90 Days =====")
    print(forecast)