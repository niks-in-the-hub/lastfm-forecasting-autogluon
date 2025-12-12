import pandas as pd
import os
import glob
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pickle


def run_training(tsv_folder_path: str) -> None:
    """
    Train a time series forecasting model using AutoGluon.

    This function locates the file in the specified directory,
    prepares it for time series modeling,
    trains multiple forecasting models using AutoGluon,
    and saves the best performing model to disk.

    The input file is expected to contain at least two columns:
      - date   : a date column
      - target : numeric value to forecast (number_of_sessions)

    Parameters:-
    tsv_folder_path : str
        Path to a folder containing one or more tab-separated CSV files.

    Returns:-
    None
        This function performs model training and serialization but does
        not return a value.
    """
    # Find latest .csv file (tab-separated)
    csv_files = glob.glob(os.path.join(tsv_folder_path, "*.csv"))
    latest_file = max(csv_files, key=os.path.getmtime)

    df = pd.read_csv(latest_file, sep='\t', parse_dates=['date'])
    df.columns = ['timestamp', 'target']
    df['item_id'] = 'series_1'

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')

    predictor = TimeSeriesPredictor(target="target", prediction_length=90, freq="D", eval_metric="RMSE")
    predictor.fit(ts_df, presets="best_quality", hyperparameters={"DeepAR": {}, "AutoARIMA": {}}, time_limit=600)

    best_model_name = predictor.model_best
    model_object = predictor._trainer.load_model(best_model_name)

    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_object, f)

    print(f"\n Training completed and best model saved as best_model.pkl")