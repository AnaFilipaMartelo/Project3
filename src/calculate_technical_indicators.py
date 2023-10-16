import pandas as pd
import numpy as np
def calculate_technical_indicators(formatted_price_data, fast_window, slow_window, rsi_window, sentiment_summary):
    """
    Calculate technical indicators and incorporate sentiment score summary as features for machine learning.
    Args:
        formatted_price_data (DataFrame): DataFrame with ticker columns and date-indexed close prices.
        fast_window (int): The window length for fast moving averages (e.g., 2 for short-term).
        slow_window (int): The window length for slow moving averages (e.g., 14 for longer-term).
        rsi_window (int): The period for calculating the Relative Strength Index (RSI).
        sentiment_summary (dict): A dictionary containing sentiment score summaries for each ticker.                   Returns:
        DataFrame: A DataFrame containing calculated technical indicators and sentiment score summaries for each ticker.
    """
    # Make a copy of the input data
    data = formatted_price_data.copy()
    # Calculate slow SMA, fast SMA, and EMA for each ticker
    for ticker in data.columns:
        data[f'SMA_Slow_{ticker}'] = data[ticker].rolling(window=slow_window).mean()
        data[f'SMA_Fast_{ticker}'] = data[ticker].rolling(window=fast_window).mean()
        data[f'EMA_{ticker}'] = data[ticker].ewm(span=slow_window, adjust=False).mean()
        # Calculate RSI (Relative Strength Index)
        percentage_change = data[ticker].pct_change()
        gain = percentage_change.where(percentage_change > 0, 0)
        loss = -percentage_change.where(percentage_change < 0, 0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        relative_strength = avg_gain / avg_loss
        rs_index = 100 - (100 / (1 + relative_strength))
        data[f'RSI_{ticker}'] = rs_index
        # Add the setiment score sum column for each tickers after the technical indicators for that ticker
        data[f'Sentiment_Score_Sum_{ticker}'] = sentiment_summary[ticker][f'{ticker}_sentiment_score_sum']
    # Drop rows with NaN values due to rolling and EMA calculations
    data.dropna(inplace=True)
    return data