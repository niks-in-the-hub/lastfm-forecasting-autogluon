import pandas as pd
from pyspark.sql import functions as F

def fill_missing_dates(df):
    """
    Ensures all dates between min and max appear in the dataframe.
    Missing dates receive number_of_sessions = 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['user_id', 'date', 'number_of_sessions']

    Returns
    -------
    pandas.DataFrame
        With all dates present for each user, missing dates filled with 0.
    """

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create full date range
    full_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    # Prepare output list
    output_frames = []

    # Process per user_id (because you may have ties)
    for user in df['user_id'].unique():
        # Filter user
        user_df = df[df['user_id'] == user].set_index('date')

        # Reindex to full date range
        user_expanded = user_df.reindex(full_range, fill_value=0)

        # Restore columns
        user_expanded.index.name = 'date'
        user_expanded = user_expanded.reset_index()

        # Add back user_id column
        user_expanded['user_id'] = user

        # Reorder columns
        user_expanded = user_expanded[['user_id', 'date', 'number_of_sessions']]

        output_frames.append(user_expanded)

    # Combine all users (handles ties)
    result = pd.concat(output_frames).reset_index(drop=True)

    # Convert date back to YYYY-MM-DD string
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')

    # Required columns
    result = result[['date', 'number_of_sessions']]

    return result



def validate_date_continuity(df):
    """
    Validates that:
      1) All dates are unique (no duplicates)
      2) All dates between the min and max are present (no missing days)

    Input:
      df: Spark DataFrame with columns ['date', 'number_of_sessions']

    Prints results directly.
    """

    # Convert date column to proper date type (safe even if already a date)
    df = df.withColumn("date", F.to_date("date"))

    # Check for duplicate dates
    duplicate_count = (
        df.groupBy("date")
          .count()
          .filter(F.col("count") > 1)
          .count()
    )

    if duplicate_count == 0:
        print(" PASS: No duplicate dates found.")
    else:
        print(f" FAIL: {duplicate_count} duplicate date(s) found!")

    # Check for missing dates
    date_stats = df.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
        F.count("date").alias("row_count")
    ).collect()[0]

    min_date = date_stats["min_date"]
    max_date = date_stats["max_date"]
    row_count = date_stats["row_count"]

    # Expected number of dates in full range
    expected_days = (max_date - min_date).days + 1

    if expected_days == row_count:
        print(" PASS: All dates from min(date) to max(date) are present.")
    else:
        missing = expected_days - row_count
        print(f" FAIL: {missing} date(s) missing between {min_date} and {max_date}.")

    # Optional summary print
    print(f"Date range: {min_date} → {max_date}")
    print(f"Rows present: {row_count}, Rows expected: {expected_days}")