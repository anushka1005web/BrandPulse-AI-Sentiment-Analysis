# BrandPulse AI — Sentiment Analysis on Twitter Data (NLP)

This project follows the supplied project specification:
- Sentiment classification: Positive / Negative / Neutral
- Classical NLP: TF-IDF + Logistic Regression
- Deep Learning: Embedding + LSTM
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Dashboard: Streamlit with prediction, sentiment distribution and trend visualization

## Project structure

```text
BrandPulse-AI-Sentiment-Analysis/
├── data/
│   ├── README.md
│   └── twitter_sentiment_sample.csv
├── models/
├── notebooks/
│   ├── 01Preprocessingand_Classical.py
│   └── 02DeepLearning_LSTM.py
├── dashboard/
│   └── app.py
├── reports/
│   └── performance_report_template.md
├── requirements.txt
├── train_classical.py
├── train_lstm.py
├── evaluate_models.py
├── utils.py
└── README.md
```

## Important dataset note

The supplied specification recommends the Sentiment140 dataset (1.6 million tweets) or Twitter US Airline Sentiment. The included CSV is only a small starter dataset so the code can be tested immediately. For the final college project, replace it with the larger dataset and convert it to the columns:

- `tweet`
- `sentiment`
- optional `timestamp`

For Sentiment140, map labels:
- 0 -> Negative
- 2 -> Neutral (if present)
- 4 -> Positive

## Setup in VS Code

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install:
```bash
pip install -r requirements.txt
```

## Run classical model

```bash
python train_classical.py
```

## Run LSTM

For a quick test:
```bash
python train_lstm.py
```

For the full Sentiment140 dataset, change `MAX_SAMPLES = None` in `train_lstm.py` only if your machine has enough RAM/GPU. The project specification notes that GPU is recommended for LSTM training.

## Evaluate

```bash
python evaluate_models.py
```

## Dashboard

After training at least the classical model:
```bash
streamlit run dashboard/app.py
```

The dashboard can:
- accept a custom tweet
- predict sentiment
- show sentiment distribution from the CSV
- show a time trend when a timestamp column is available

## Expected final deliverables

1. Classical NLP implementation
2. LSTM implementation
3. Streamlit dashboard
4. Performance comparison report
5. Confusion matrices
