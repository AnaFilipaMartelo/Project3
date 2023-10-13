import pandas as pd

def format_price_data(data_df):
    """Formats the given data DataFrame to a specific format by pivoting the table and organizing the columns. 
    
    Args:
        data (DataFrame): The DataFrame containing the data to be restructured.

    Returns:
        DataFrame: The restructured DataFrame, where each ticker symbol corresponds to a column,
               and the rows represent dates with corresponding close prices.
    """
    
    # Reset index 
    data_df = data_df.reset_index()

    # Convert timestamp to datetime and extract date and hour+minute
    data_df['Date'] = pd.to_datetime(data_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

    # Convert the 'Date' column back to a datetime format
    data_df['Date'] = pd.to_datetime(data_df['Date'])

    # Pivot table to have ticker symbols as columns and date as index
    formatted_df = data_df.pivot(index='Date', columns='symbol', values='close')    
    
    return formatted_df