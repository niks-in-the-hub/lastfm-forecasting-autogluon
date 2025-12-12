from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


# Schema definition
LASTFM_SCHEMA = StructType([
    StructField("user_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("artist_id", StringType(), True),
    StructField("artist_name", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("track_name", StringType(), True),
])

# SparkSession builder
def create_spark(app_name: str = "LastFM ETL"):
    """
    Creates (or returns existing) SparkSession with consistent configuration.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")  # disable broadcast
        .config("spark.sql.adaptive.enabled", False)           # avoid shuffle rewrite
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    # To reduce log noise
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# TSV to Parquet Conversion
def tsv_to_parquet(input_path: str, output_path: str):
    """
    Reads a TSV file using the predefined LASTFM_SCHEMA
    and writes it as a Parquet dataset.
    """
    spark = create_spark("TSV to Parquet Conversion")

    df = (
        spark.read
        .option("header", "false")
        .option("delimiter", "\t")
        .schema(LASTFM_SCHEMA)
        .csv(input_path)
    )

    df.write.mode("overwrite").parquet(output_path)

    print(f"TSV -> Parquet conversion complete!\n   Output folder: {output_path}")


# Inspect parquet folder before creating df
def inspect_parquet_folder(input_path: str):
    """
    Loads parquet files into a DataFrame,
    prints schema + sample rows,
    and returns (df, row_count, col_count).
    """
    spark = create_spark("Inspect Parquet Folder")

    print(f"\n Loading Parquet data from: {input_path}")

    df = spark.read.parquet(input_path)

    print("\n===== DATAFRAME SCHEMA =====")
    df.printSchema()

    print("\n===== SAMPLE ROWS =====")
    df.show(10, truncate=False)

    row_count = df.count()
    col_count = len(df.columns)

    print("\n===== SUMMARY =====")
    print(f"Rows: {row_count}")
    print(f"Columns: {col_count}\n")

    return df, row_count, col_count