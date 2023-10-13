import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
import time

def collect_sentiment_data(tickers, time_period):
    """
    Retrieve news data for a list of tickers in a defined time period.

    Args:
        tickers (list): List of tickers that we want to collect data.
        time_period (tuple): Tuple containing the start and end date of the historical data.

    Returns:
        A DataFrame with the news data for the tickers.

    """
    # Load environment variables from the .env file
    load_dotenv()

    # Set up Alpaca API credentials and base URL
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    api = REST(key_id=alpaca_api_key, secret_key=alpaca_secret_key, base_url="https://paper-api.alpaca.markets")

    # Define the date range (start and end dates) 
    start_date, end_date = time_period
    date_format = "%Y-%m-%d"

    # Convert the date type to 'datetime' since it is a string type
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    # Create an empty DataFrame to store news data
    news_df = pd.DataFrame(columns=['Ticker', 'Date', 'Title'])

    # Inintialize the variable that will be counting the days the loop goes through
    x=0

    # Loop through tickers
    for ticker in tickers:
        # Loop through each day within the date range
        current_date = start_date
        while current_date <= end_date:
            # Define the date range for the current day (start and end times) in ISO 8601 format
            # Collect news during U.S. market hours from 9:30 AM to 4:00 PM Eastern Time
            current_day_start = current_date.replace(hour=9, minute=30, second=0, microsecond=0).isoformat() + "Z"
            current_day_end = current_date.replace(hour=16, minute=59, second=59, microsecond=999999).isoformat() + "Z"

            # Retrieve news articles for the current day and ticker
            news_articles = api.get_news(symbol=ticker, start=current_day_start, end=current_day_end)
        
            # Append news data to the DataFrame
            for article in news_articles:
                news_df = news_df.append({
                    'Ticker': ticker,
                    'Date': article.created_at,
                    'Title': article.headline  
                }, ignore_index=True)
        
            # Move to the next day
            current_date += timedelta(days=1)

            # Increment the day counter for every loop iteration
            x+=1
            # Sleep every 90 days to prevent API overloads 
            if x%90==0:
                time.sleep(15)     
            # Sleep for an extra 30 seconds after 1 year of data retrieval 
            if x%365==0:
                time.sleep(30)
            # Sleep for an extra 2 minutes after retrieving almost 10 years of data
            if x%3640==0:
                time.sleep(120)
          
    return news_df
