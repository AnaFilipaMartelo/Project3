import pandas as pd 

def create_sentiment_summary(tickers, sentiment_df, formatted_price_data):
    """
    Create a sentiment summary in 3 steps:
    Step 1: Calculate daily sentiment score statistics (mean, min, max, sum) for each ticker and date.
    Step 2: Resample sentiment data to an hourly frequency and forward-fill to prepare for merging.
    Step 3: Merge hourly sentiment statistics with close price data and store in a dictionary.

    Args:
        tickers (list): List of tickers to collect data for.
        sentiment_df (DataFrame): DataFrame containing sentiment data.
        formatted_price_data (DataFrame): DataFrame with ticker columns and date-indexed close prices.
        
    Returns:
        A dictionary with DataFrames containing sentiment statistics and close prices for each ticker.
    """
    
    # Step 1: Data Preparation and Calculation
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])  # Convert 'Date' to datetime
    sentiment_df.set_index('Date', inplace=True)  # Set 'Date' as the DataFrame index
    sentiment_summaries = {}  # To store daily sentiment statistics

    # Calculate daily mean sentiment score for all tickers
    daily_mean_sentiment = sentiment_df.groupby(['Ticker', sentiment_df.index.date])['Sentiment_Score'].mean()
    # Calculate daily sum sentiment score for all tickers
    daily_sum_sentiment = sentiment_df.groupby(['Ticker', sentiment_df.index.date])['Sentiment_Score'].sum()
    # Calculate daily max sentiment score for all tickers
    daily_max_sentiment = sentiment_df.groupby(['Ticker', sentiment_df.index.date])['Sentiment_Score'].max()
    # Calculate daily min sentiment score for all tickers
    daily_min_sentiment = sentiment_df.groupby(['Ticker', sentiment_df.index.date])['Sentiment_Score'].min()

    for ticker in tickers:
        # Create sentiment summary DataFrame for the current ticker
        sentiment_summary = pd.DataFrame({
            f'{ticker}_sentiment_score_mean': daily_mean_sentiment[ticker],
            f'{ticker}_sentiment_score_sum': daily_sum_sentiment[ticker],
            f'{ticker}_sentiment_score_max': daily_max_sentiment[ticker],
            f'{ticker}_sentiment_score_min': daily_min_sentiment[ticker]
        })

        sentiment_summaries[ticker] = sentiment_summary

    # Step 2: Resampling and Forward-Filling
    hourly_sentiment_summaries = {} # To store hourly sentiment summaries

    for ticker, summary_df in sentiment_summaries.items():
        # Convert the index to a datetime index
        summary_df.index = pd.to_datetime(summary_df.index)

        # Resample to hourly frequency and forward-fill
        hourly_summary = summary_df.resample('H').ffill()
        
        # Store the hourly summary in the dictionary
        hourly_sentiment_summaries[ticker] = hourly_summary

    # Step 3: Merging with Price Data
    merged_data = {}  # To store the merged DataFrames for each ticker

    for ticker in tickers:
        # Merge the hourly sentiment summary with the corresponding price data on the index (Date)
        merged_data[ticker] = hourly_sentiment_summaries[ticker].merge(
            formatted_price_data[ticker], left_index=True, right_index=True, how='left'
        )

        # Drop rows with NaN values (rows without corresponding price data)
        merged_data[ticker] = merged_data[ticker].dropna()

    return merged_data
