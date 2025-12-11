import os
from utils import tsv_to_parquet, inspect_parquet_folder
from data_generation import data_generation
from pre_process import fill_missing_dates

from pyspark.sql import SparkSession



def main():
    """
    Main ETL + analysis pipeline for LastFM dataset.
    """
    spark = SparkSession.builder \
        .appName("ML Challenge") \
        .getOrCreate()

    #Define paths
    tsv_input_path = "/Users/nikkitharamesh/Desktop/ml challenge/data/userid-timestamp-artid-artname-traid-traname.tsv"
    parquet_output_path = "/Users/nikkitharamesh/Desktop/ml challenge/output/parquet_out"
    tsv_final_output = "/Users/nikkitharamesh/Desktop/ml challenge/output/result_file"

    if not tsv_input_path:
        raise ValueError(
            "ERROR: Input file not found"
        )

    print(f"\n=== Running with configuration ===")
    print(f"Input TSV:       {tsv_input_path}")
    print(f"Parquet Output:  {parquet_output_path}")
    print(f"Final TSV Output:{tsv_final_output}\n")

    # Convert TSV to Parquet
    tsv_to_parquet(tsv_input_path, parquet_output_path)

    # Load parquet dataset
    df, row_count, col_count = inspect_parquet_folder(parquet_output_path)

    df.show(5)

    # Final analysis
    print("\n Top 1 user ")
    generated_df = data_generation(df)

    pre_processed_df = fill_missing_dates(generated_df.toPandas())
    # Convert back to Spark
    pre_processed_df = spark.createDataFrame(pre_processed_df)

    pre_processed_df.show(10)

    # Save results as TSV
    (
        pre_processed_df.coalesce(1).write \
            .mode("overwrite") \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .csv(tsv_final_output)
    )

    print(f"\n Final results written to: {tsv_final_output}\n")


if __name__ == "__main__":
    main()