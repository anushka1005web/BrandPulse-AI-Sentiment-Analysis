import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import clean_text


# ============================================================
# SENTIMENT140 DATASET PATH
# ============================================================

DATA_PATH = "data/training.1600000.processed.noemoticon.csv"

# Model folder
MODEL_DIR = "models"

# Create models folder
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD SENTIMENT140 DATASET
# ============================================================

print("Loading Sentiment140 dataset...")

columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# ============================================================
# CONVERT SENTIMENT LABELS
# 0 = Negative
# 4 = Positive
# ============================================================

df["sentiment"] = df["sentiment"].map({
    0: "Negative",
    4: "Positive"
})

# Remove missing values
df = df.dropna(subset=["sentiment", "text"])


# ============================================================
# CLEAN TEXT
# ============================================================

print("Cleaning tweets...")

df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)

print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

print("Making predictions...")

pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, pred)

print("\n========================================")
print("CLASSICAL NLP RESULTS")
print("========================================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    f"{MODEL_DIR}/classical_model.pkl"
)

joblib.dump(
    vectorizer,
    f"{MODEL_DIR}/tfidf_vectorizer.pkl"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": pred
})

prediction_df.to_csv(
    f"{MODEL_DIR}/classical_predictions.csv",
    index=False
)


print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("classical_model.pkl")
print("tfidf_vectorizer.pkl")
print("classical_predictions.csv")