import re
import pandas as pd

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def load_data(path="data/twitter_sentiment.csv"):
    df = pd.read_csv(path)
    if "tweet" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'tweet' and 'sentiment' columns.")
    df = df.dropna(subset=["tweet", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()
    # Common Sentiment140 numeric labels
    mapping = {"0":"Negative", "2":"Neutral", "4":"Positive"}
    df["sentiment"] = df["sentiment"].replace(mapping)
    df["clean_text"] = df["tweet"].apply(clean_text)
    return df
