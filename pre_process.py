import pandas as pd

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