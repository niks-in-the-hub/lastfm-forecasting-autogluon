from pyspark.sql import functions as F
from pyspark.sql.window import Window

def data_generation(df):

    spark = df.sparkSession
    spark.conf.set("spark.sql.shuffle.partitions", "50")   # reduce shuffle load
    spark.conf.set("spark.sql.adaptive.enabled", True)     # let Spark optimize

    # 1: Parse timestamps
    df_parsed = df.withColumn(
        "ts", F.to_timestamp("timestamp")
    ).withColumn(
        "date", F.to_date("ts")
    )

    # 2: Window (safe because per user, but still expensive)
    w = Window.partitionBy("user_id").orderBy("ts")

    df_with_diff = df_parsed.withColumn(
        "prev_ts", F.lag("ts").over(w)
    ).withColumn(
        "diff_minutes", 
        (F.col("ts").cast("long") - F.col("prev_ts").cast("long"))/60
    )

    # 3: New session flag
    df_sessionized = df_with_diff.withColumn(
        "is_new_session",
        F.when(F.col("prev_ts").isNull(), 1)
         .when(F.col("diff_minutes") > 20, 1)
         .otherwise(0)
    )

    # 4: Cumulative sum session_id
    df_sessionized = df_sessionized.withColumn(
        "session_id",
        F.sum("is_new_session").over(w)
    )

    # 5: Sessions per user
    user_session_counts = (
        df_sessionized
        .select("user_id", "session_id").distinct()
        .groupBy("user_id")
        .count()
        .withColumnRenamed("count", "total_sessions")
    )

    # *** NEW: Compute max sessions WITHOUT .first() ***
    max_sessions_df = user_session_counts.agg(F.max("total_sessions").alias("max_sessions"))
    
    # join instead of collect()
    top_users = user_session_counts.join(
        max_sessions_df,
        on=F.col("total_sessions") == F.col("max_sessions"),
        how="inner"
    ).select("user_id").distinct()

    # 7: Filter only top users
    df_top = df_sessionized.join(top_users, "user_id")

    # 8: daily session aggregation (no global sort)
    daily_sessions = (
        df_top.select("user_id", "date", "session_id")
        .distinct()
        .groupBy("user_id", "date")
        .agg(F.count("*").alias("number_of_sessions"))
        .orderBy("user_id", "date")
    )

    return daily_sessions
