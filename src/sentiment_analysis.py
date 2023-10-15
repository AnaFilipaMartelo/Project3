from transformers import pipeline
import pandas as pd

# Initialize the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def perform_sentiment_analysis(news_data):
    """
    Analyzes the sentiment of news headlines using a pre-trained NLP model to calculate sentiment scores and labels.

    Args:
        news_data (DataFrame): DataFrame containing news data with 'Ticker', 'Date', and 'Title' columns.

    Returns:
        DataFrame: DataFrame with sentiment analysis results including 'Ticker', 'Date', 'Title',
        'Sentiment_Score', and 'Sentiment_Label' columns.
    """
    
    # Create an empty DataFrame to store sentiment analysis results
    sentiment_df = pd.DataFrame(columns=['Ticker', 'Date', 'Title', 'Sentiment_Score', 'Sentiment_Label'])

    # Loop through each row in news_data
    for index, row in news_data.iterrows():
        ticker = row['Ticker']
        date = row['Date']
        title = row['Title']

        # Perform sentiment analysis
        sentiment_result = sentiment_analyzer(title)[0]

        # Extract sentiment score and label
        sentiment_score = sentiment_result['score']
        sentiment_label = sentiment_result['label']

        # Append results to sentiment_df
        sentiment_df = sentiment_df.append({
            'Ticker': ticker,
            'Date': date,
            'Title': title,
            'Sentiment_Score': sentiment_score,
            'Sentiment_Label': sentiment_label
        }, ignore_index=True)

    return sentiment_df