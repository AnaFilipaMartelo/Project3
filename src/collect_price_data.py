import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas as pd

def collect_price_data(tickers, time_period):
    """
    Access historical data for a list of tickers in a defined time period.

    Args:
        tickers (list): List of tickers that we want to collect data.
        time_period (tuple): Tuple containing the start and end date of the historical data.

    Returns:
        A DataFrame with the historical price data for the tickers.

    """

    # Load the environment variables from the .env file
    load_dotenv()

    # Set the variables for the Alpaca API and secret keys
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

    # Create the Alpaca tradeapi.REST object
    alpaca = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version="v2")

    # Set the timeframe for the data to 1 hour
    timeframe = "1H"

    # Use Alpaca get_barset function to get the data for the tickers
    tickers_price_data = alpaca.get_bars(
        tickers,
        timeframe,
        start = time_period[0],
        end = time_period[1]
    ).df

    return tickers_price_data 