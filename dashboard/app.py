import streamlit as st
import pandas as pd
import joblib
import re
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main title */
h1 {
    font-weight: 800 !important;
    color: #172033;
}

h2 {
    font-weight: 750 !important;
    color: #172033;
}

h3 {
    font-weight: 700 !important;
    color: #27364d;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.06);
}

div[data-testid="stMetricLabel"] {
    color: #667085 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #4f46e5 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 700;
    padding: 12px;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 7px 18px rgba(79,70,229,0.25);
}

/* Text area */
textarea {
    border-radius: 12px !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Divider */
hr {
    margin-top: 30px;
    margin-bottom: 30px;
}

/* Info boxes */
.info-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

.info-title {
    font-size: 18px;
    font-weight: 700;
    color: #344054;
}

.info-text {
    color: #667085;
    margin-top: 8px;
}

/* Hero */
.hero-box {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 20px;
    padding: 35px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 12px 30px rgba(79,70,229,0.20);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 8px;
    opacity: 0.92;
}

/* Small badge */
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    margin-bottom: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #667085;
    padding: 25px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "classical_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison.csv"
)

CLASSICAL_CM = os.path.join(
    BASE_DIR,
    "reports",
    "classical_confusion_matrix.png"
)

LSTM_CM = os.path.join(
    BASE_DIR,
    "reports",
    "lstm_confusion_matrix.png"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📊 BrandPulse AI")

    st.caption(
        "AI-Powered Twitter Sentiment Analysis"
    )

    st.divider()

    st.markdown("### 📌 Project")

    st.write("Natural Language Processing")
    st.write("Machine Learning")
    st.write("Sentiment Classification")

    st.divider()

    st.markdown("### 🤖 Models")

    st.write("TF-IDF + Logistic Regression")
    st.write("LSTM Neural Network")

    st.divider()

    st.markdown("### 📂 Dataset")

    st.write("Sentiment140")
    st.write("1.6 Million Tweets")

    st.divider()

    st.caption(
        "BrandPulse AI\nCollege Project"
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero-box">

<div class="badge">
AI • NLP • MACHINE LEARNING
</div>

<div class="hero-title">
📊 BrandPulse AI
</div>

<div class="hero-subtitle">
Twitter Sentiment Analysis using Natural Language Processing
and Machine Learning
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.subheader("📈 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="📂 Dataset Size",
        value="1.6M",
        delta="Tweets"
    )

with col2:

    st.metric(
        label="🤖 Classical NLP",
        value="81.89%",
        delta="Accuracy"
    )

with col3:

    st.metric(
        label="🧠 LSTM",
        value="78.10%",
        delta="Accuracy"
    )

with col4:

    st.metric(
        label="😊 Sentiments",
        value="2",
        delta="Positive / Negative"
    )


# =========================================================
# TWEET ANALYSIS
# =========================================================

st.divider()

st.subheader("📝 Analyze Tweet")

st.write(
    "Enter a tweet below and let the trained "
    "Machine Learning model classify its sentiment."
)

tweet = st.text_area(
    "Tweet",
    placeholder=(
        "Example: I really love this product! "
        "The experience is amazing."
    ),
    height=130,
    label_visibility="collapsed"
)

analyze_button = st.button(
    "🔍 Analyze Sentiment"
)


# =========================================================
# PREDICTION
# =========================================================

if analyze_button:

    if tweet.strip() == "":

        st.warning(
            "⚠️ Please enter a tweet before analyzing."
        )

    else:

        cleaned = clean_text(tweet)

        vector = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            vector
        )[0]

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        if prediction == "Positive":

            st.success(
                "😊 POSITIVE SENTIMENT"
            )

            st.write(
                "The model classified this tweet "
                "as **Positive**."
            )

        elif prediction == "Negative":

            st.error(
                "😞 NEGATIVE SENTIMENT"
            )

            st.write(
                "The model classified this tweet "
                "as **Negative**."
            )

        else:

            st.info(
                "😐 NEUTRAL SENTIMENT"
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader(
    "📊 Model Performance Comparison"
)

if os.path.exists(COMPARISON_PATH):

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    accuracy_row = comparison[
        comparison["Metric"] == "Accuracy"
    ]

    if not accuracy_row.empty:

        classical_accuracy = (
            accuracy_row[
                "Classical NLP"
            ].iloc[0] * 100
        )

        lstm_accuracy = (
            accuracy_row[
                "LSTM"
            ].iloc[0] * 100
        )

        chart_data = pd.DataFrame(
            {
                "Accuracy (%)": [
                    classical_accuracy,
                    lstm_accuracy
                ]
            },
            index=[
                "Classical NLP",
                "LSTM"
            ]
        )

        st.write(
            "### Accuracy Comparison"
        )

        st.bar_chart(
            chart_data
        )

else:

    st.warning(
        "Model comparison file not found."
    )


# =========================================================
# CONFUSION MATRICES
# =========================================================

st.divider()

st.subheader(
    "📉 Confusion Matrices"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "### 🤖 Classical NLP"
    )

    if os.path.exists(CLASSICAL_CM):

        st.image(
            CLASSICAL_CM,
            use_container_width=True
        )

    else:

        st.warning(
            "Classical confusion matrix not found."
        )


with col2:

    st.markdown(
        "### 🧠 LSTM"
    )

    if os.path.exists(LSTM_CM):

        st.image(
            LSTM_CM,
            use_container_width=True
        )

    else:

        st.warning(
            "LSTM confusion matrix not found."
        )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.divider()

st.subheader(
    "📂 Dataset Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">📚 Dataset</div>'
        '<div class="info-text">'
        'Sentiment140'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🐦 Tweets</div>'
        '<div class="info-text">'
        '1,600,000 Tweets'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">😊 Classes</div>'
        '<div class="info-text">'
        'Positive & Negative'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# TECHNOLOGIES
# =========================================================

st.divider()

st.subheader(
    "🛠️ Technologies Used"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🐍 Python</div>'
        '<div class="info-text">'
        'Programming'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🧠 NLP</div>'
        '<div class="info-text">'
        'Text Processing'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🤖 Machine Learning</div>'
        '<div class="info-text">'
        'Logistic Regression'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🔥 Deep Learning</div>'
        '<div class="info-text">'
        'LSTM'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.divider()

st.subheader(
    "📌 Project Summary"
)

st.info(
    "BrandPulse AI analyzes Twitter text and classifies "
    "sentiment using NLP and Machine Learning techniques. "
    "The project compares a classical TF-IDF + Logistic "
    "Regression approach with an LSTM-based deep learning model."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

<b>BrandPulse AI</b><br>

Twitter Sentiment Analysis using NLP<br>

Built with Python • Machine Learning • LSTM • Streamlit

</div>
""", unsafe_allow_html=True)