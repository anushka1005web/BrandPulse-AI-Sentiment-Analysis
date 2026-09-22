import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from utils import clean_text


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "data/training.1600000.processed.noemoticon.csv"
MODEL_DIR = "models"

MAX_SAMPLES = 100000
MAX_WORDS = 30000
MAX_LEN = 80

EPOCHS = 8
BATCH_SIZE = 128

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

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


# ==========================================
# CONVERT LABELS
# ==========================================

# Sentiment140:
# 0 = Negative
# 4 = Positive

df["sentiment"] = df["sentiment"].map({
    0: "Negative",
    4: "Positive"
})

df = df.dropna(
    subset=["sentiment", "text"]
)


# ==========================================
# SAMPLE DATA
# ==========================================

if MAX_SAMPLES and len(df) > MAX_SAMPLES:

    print(
        f"Using {MAX_SAMPLES} tweets for LSTM training..."
    )

    df = df.sample(
        MAX_SAMPLES,
        random_state=42
    )

print(
    "Tweets used:",
    len(df)
)


# ==========================================
# CLEAN TEXT
# ==========================================

print("Cleaning tweets...")

df["clean_text"] = df["text"].apply(
    clean_text
)

df = df[
    df["clean_text"].str.strip() != ""
]


# ==========================================
# LABEL ENCODING
# ==========================================

label_map = {
    "Negative": 0,
    "Positive": 1
}

X = df["clean_text"].values

y = df["sentiment"].map(
    label_map
).values


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("Splitting dataset...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training samples:",
    len(X_train_text)
)

print(
    "Testing samples:",
    len(X_test_text)
)


# ==========================================
# TOKENIZATION
# ==========================================

print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    X_train_text
)


# ==========================================
# CONVERT TEXT TO SEQUENCES
# ==========================================

print("Converting text to sequences...")

X_train = tokenizer.texts_to_sequences(
    X_train_text
)

X_test = tokenizer.texts_to_sequences(
    X_test_text
)


# ==========================================
# PADDING
# ==========================================

print("Padding sequences...")

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN,
    padding="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LEN,
    padding="post"
)


# ==========================================
# BUILD BIDIRECTIONAL LSTM
# ==========================================

print("Building Bidirectional LSTM model...")

model = Sequential([

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        mask_zero=True
    ),

    Bidirectional(
        LSTM(128)
    ),

    Dropout(0.4),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        2,
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# ==========================================
# TRAIN
# ==========================================

print("\n========================================")
print("TRAINING BIDIRECTIONAL LSTM")
print("========================================")

history = model.fit(

    X_train,
    y_train,

    validation_split=0.10,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ],

    verbose=1
)


# ==========================================
# PREDICTION
# ==========================================

print("\nMaking predictions...")

probs = model.predict(
    X_test,
    verbose=0
)

pred = np.argmax(
    probs,
    axis=1
)


# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    pred
)

print("\n========================================")
print("LSTM RESULTS")
print("========================================")

print(
    f"LSTM Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving LSTM model...")

model.save(
    f"{MODEL_DIR}/lstm_model.keras"
)


# ==========================================
# SAVE TOKENIZER
# ==========================================

with open(
    f"{MODEL_DIR}/tokenizer.json",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        tokenizer.to_json()
    )


# ==========================================
# SAVE CONFIG
# ==========================================

with open(
    f"{MODEL_DIR}/lstm_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "max_words": MAX_WORDS,
            "max_len": MAX_LEN
        },
        f
    )


# ==========================================
# SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({

    "actual": y_test,

    "predicted": pred

})

prediction_df.to_csv(
    f"{MODEL_DIR}/lstm_predictions.csv",
    index=False
)


# ==========================================
# COMPLETE
# ==========================================

print("\n========================================")
print("LSTM MODEL SAVED SUCCESSFULLY")
print("========================================")

print("lstm_model.keras")
print("tokenizer.json")
print("lstm_config.json")
print("lstm_predictions.csv")