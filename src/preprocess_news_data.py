import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_news_data(news_df):
    """
    Preprocesses the news data by applying common NLP text preprocessing steps.
    This function performs the following preprocessing steps on the news headlines:
        1.	Convert text to lowercase.
        2.	Remove special characters and digits.
        3.	Tokenization.
        4.	Stopword removal.
        5.	Lemmatization.

    Args:
        news_df (pd.DataFrame): DataFrame containing news data with 'Title' column.

    Returns:
        DataFrame: Preprocessed news data.
    """
    # Load stopwords and initialize the WordNet lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Preprocessing steps for each news headline
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r"[^a-z ]", "", text)
        # Tokenization and stop word removal
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return " ".join(tokens)

    # Apply preprocessing to the 'Title' column
    news_df["Title"] = news_df["Title"].apply(preprocess_text)

    return news_df