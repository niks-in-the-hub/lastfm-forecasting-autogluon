import os
from utils import tsv_to_parquet, inspect_parquet_folder
from data_generation import data_generation
from pre_process import fill_missing_dates, validate_date_continuity
from train import run_training
from predict import run_forecasting
from pyspark.sql import SparkSession
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PySpark local job")
    
    parser.add_argument(
         "-i", "--input",
        required=True,
        help="Path to input tsv file "
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        default=os.getcwd(),
        help="Path to output folder "
    )

    return parser.parse_args()


def main():
    """
    Main ETL + analysis pipeline for LastFM dataset.
    """
    spark = SparkSession.builder \
        .appName("ML Challenge") \
        .getOrCreate()

    args = parse_args()

    tsv_input_path = args.input
    tsv_training_data_path = args.output
    forecast_results_output_path = os.path.join(tsv_training_data_path,"forecast_results")
    os.makedirs(forecast_results_output_path, exist_ok=True)

    # Parquet directory inside output folder
    parquet_path = os.path.join(tsv_training_data_path, "parquet")
    os.makedirs(parquet_path, exist_ok=True)


    if not tsv_input_path:
        raise ValueError(
            "ERROR: Input file not found"
        )

    print(f"\n=== Running with configuration ===")
    print(f"Input TSV:       {tsv_input_path}")
    print(f"Parquet Output:  {parquet_path}")
    print(f"Final TSV Output:{tsv_training_data_path}\n")

    # Convert TSV to Parquet
    tsv_to_parquet(tsv_input_path, parquet_path)

    # Load parquet dataset
    df, row_count, col_count = inspect_parquet_folder(parquet_path)

    df.show(5)

    # Final analysis
    print("\n Top 1 user ")
    generated_df = data_generation(df)

    pre_processed_df = fill_missing_dates(generated_df.toPandas())
    # Convert back to Spark
    pre_processed_df = spark.createDataFrame(pre_processed_df)

    pre_processed_df.show(10)

    validate_date_continuity(pre_processed_df)

    # Save results as TSV in a subdirectory to avoid overwriting forecast_results
    csv_output_path = os.path.join(tsv_training_data_path, "processed_data")
    (
        pre_processed_df.coalesce(1).write \
            .mode("overwrite") \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .csv(csv_output_path)
    )

    print(f"\n Final results written to: {csv_output_path}\n")

    # Run training and forecasting using TSV outputs
    run_training(csv_output_path)
    run_forecasting(csv_output_path, forecast_results_output_path)

if __name__ == "__main__":
    main()