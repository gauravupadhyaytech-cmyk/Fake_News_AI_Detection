import streamlit as st
import joblib
import numpy as np
import pickle
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# NLTK DATA DOWNLOAD
# ============================================================================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fake News Detection System v3.0",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2f4a 0%, #0f1d2d 100%);
    }
    h1, h2, h3 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 212, 255, 0.2);
    }
    body { color: #e0e0e0; }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.4);
    }
    [data-testid="metric-container"] {
        background: rgba(0, 212, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button { color: #00d4ff; }
    /* PDF download buttons */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LABEL ENCODING - EXACT FROM TRAINING
# ============================================================================
"""
⚠️ IMPORTANT - LABEL ENCODING (From Training):
    0 = REAL NEWS ✅
    1 = FAKE NEWS 🚨
"""

# ============================================================================
# LOAD ML MODELS
# ============================================================================
@st.cache_resource
def load_ml_models():
    try:
        current_dir     = os.path.dirname(os.path.abspath(__file__))
        model_dir       = os.path.join(current_dir, "V2_Saved_Models_ML")

        if not os.path.exists(model_dir):
            return {"status": "error", "message": f"Model directory not found at: {model_dir}"}

        lr_path         = os.path.join(model_dir, "logistic_regression.pkl")
        svc_path        = os.path.join(model_dir, "svc.pkl")
        voting_path     = os.path.join(model_dir, "voting_classifier_86acc.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        for path, name in [
            (lr_path,         "Logistic Regression"),
            (svc_path,        "SVC"),
            (voting_path,     "Voting Classifier"),
            (vectorizer_path, "TF-IDF Vectorizer"),
        ]:
            if not os.path.exists(path):
                return {"status": "error", "message": f"{name} not found at: {path}"}

        return {
            "lr":         joblib.load(lr_path),
            "svc":        joblib.load(svc_path),
            "voting":     joblib.load(voting_path),
            "vectorizer": joblib.load(vectorizer_path),
            "status":     "success",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# LOAD DL MODELS
# ============================================================================
@st.cache_resource
def load_dl_models():
    """
    Load BiLSTM M1 and DistilBERT.
    tensorflow.python.data.experimental error is caused by TF version mismatch.
    Fix: use tf.keras compat import and suppress experimental API errors.
    """
    try:
        import tensorflow as tf

        # ── Suppress experimental-ops import error ──────────────────────────
        # Some TF versions raise ModuleNotFoundError for
        # tensorflow.python.data.experimental.ops.iterator_model_ops
        # when loading a .keras model saved on a different TF version.
        # The workaround is to use tf.keras.models.load_model directly
        # (which uses the same backend) and catch any non-fatal import warnings.
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        from tensorflow import keras  # use tf.keras — avoids standalone keras conflicts

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dl_dir      = os.path.join(current_dir, "V2_dl_model_saved")

        if not os.path.exists(dl_dir):
            return {"status": "error", "message": f"DL model directory not found: {dl_dir}"}

        m1_path        = os.path.join(dl_dir, "bi-direct_m1.keras")
        tokenizer_path = os.path.join(dl_dir, "tokenizer.pkl")

        for path, name in [(m1_path, "BiLSTM M1"), (tokenizer_path, "Keras Tokenizer")]:
            if not os.path.exists(path):
                return {"status": "error", "message": f"{name} not found at: {path}"}

        # Load BiLSTM using tf.keras to avoid experimental API import errors
        bilstm_m1 = keras.models.load_model(m1_path)

        with open(tokenizer_path, "rb") as f:
            keras_tokenizer = pickle.load(f)

        # DistilBERT — optional
        bert_dir             = os.path.join(dl_dir, "distilbert_m1")
        distilbert_model     = None
        distilbert_tokenizer = None
        bert_status          = "not_found"

        if os.path.exists(bert_dir):
            try:
                from transformers import (
                    DistilBertTokenizerFast,
                    DistilBertForSequenceClassification,
                )
                distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
                distilbert_model     = DistilBertForSequenceClassification.from_pretrained(bert_dir)
                distilbert_model.eval()
                bert_status = "success"
            except Exception as e:
                bert_status = f"error: {str(e)}"

        return {
            "bilstm_m1":             bilstm_m1,
            "keras_tokenizer":       keras_tokenizer,
            "distilbert_model":      distilbert_model,
            "distilbert_tokenizer":  distilbert_tokenizer,
            "bert_status":           bert_status,
            "status":                "success",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# LOAD ALL MODELS
# ============================================================================
ml_dict = load_ml_models()
dl_dict = load_dl_models()

# ML must succeed
if ml_dict.get("status") == "error":
    st.error(f"❌ Model Loading Error: {ml_dict.get('message')}")
    st.info("""
    **Make sure your folder structure is correct:**
    ```
    FAKE NEWS PROJECT FINAL/
    ├── app.py
    ├── V2_Saved_Models_ML/
    │   ├── logistic_regression.pkl
    │   ├── svc.pkl
    │   ├── tfidf_vectorizer.pkl
    │   └── voting_classifier_86acc.pkl
    └── V2_dl_model_saved/
        ├── bi-direct_m1.keras
        ├── tokenizer.pkl
        └── distilbert_m1/
    ```
    """)
    st.stop()

lr_model     = ml_dict["lr"]
svc_model    = ml_dict["svc"]
voting_model = ml_dict["voting"]
vectorizer   = ml_dict["vectorizer"]

dl_available   = dl_dict.get("status") == "success"
bert_available = False

if dl_available:
    bilstm_m1            = dl_dict["bilstm_m1"]
    keras_tokenizer      = dl_dict["keras_tokenizer"]
    distilbert_model     = dl_dict["distilbert_model"]
    distilbert_tokenizer = dl_dict["distilbert_tokenizer"]
    bert_available       = dl_dict["bert_status"] == "success"
else:
    st.sidebar.warning(f"⚠️ DL Models: {dl_dict.get('message', 'Not loaded')}")

st.sidebar.success("✅ All ML models loaded successfully!")
if dl_available:
    st.sidebar.success("✅ BiLSTM M1 loaded!")
    if bert_available:
        st.sidebar.success("✅ DistilBERT loaded!")
    else:
        st.sidebar.warning(f"⚠️ DistilBERT: {dl_dict.get('bert_status', 'not found')}")

# ============================================================================
# MODEL LIST
# ============================================================================
ALL_MODELS = ["Logistic Regression", "SVC", "Voting Classifier"]
if dl_available:
    ALL_MODELS.append("BiLSTM (with NLP)")
    if bert_available:
        ALL_MODELS.append("DistilBERT")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
class MLPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_text(self, text):
        cleaned          = self.clean_text(text)
        tokens           = word_tokenize(cleaned)
        orig_count       = len(tokens)
        processed        = [self.lemmatizer.lemmatize(t)
                            for t in tokens
                            if t not in self.stop_words and len(t) > 2]
        return {
            "text":              ' '.join(processed),
            "final_tokens":      len(processed),
            "original_tokens":   orig_count,
            "stopwords_removed": orig_count - len(processed),
        }


class BiLSTMPreprocessor:
    """advanced_clean — exact match to training Cell 15-18"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text    = text.lower()
        text    = re.sub(r'https?://\S+|www\.\S+', '', text)
        text    = re.sub(r'<.*?>', '', text)
        text    = re.sub(r'[^a-z\s]', '', text)
        tokens  = text.split()
        cleaned = [self.lemmatizer.lemmatize(w)
                   for w in tokens if w not in self.stop_words]
        return {
            "text":              " ".join(cleaned),
            "final_tokens":      len(cleaned),
            "original_tokens":   len(tokens),
            "stopwords_removed": len(tokens) - len(cleaned),
        }


ml_preprocessor     = MLPreprocessor()
bilstm_preprocessor = BiLSTMPreprocessor()

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_logistic_regression(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    proba  = lr_model.predict_proba(vector)[0]
    pred   = int(lr_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


def predict_svc(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    proba  = svc_model.predict_proba(vector)[0]          # CalibratedClassifierCV
    pred   = int(svc_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


def predict_voting_classifier(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    try:
        proba = voting_model.predict_proba(vector)[0]
    except Exception:
        p     = int(voting_model.predict(vector)[0])
        proba = np.array([1 - p, p])
    pred = int(voting_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


def predict_bilstm(text):
    """BiLSTM M1 — advanced_clean, vocab=10000, maxlen=300, padding='post'"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    prep   = bilstm_preprocessor.preprocess_text(text)
    seq    = keras_tokenizer.texts_to_sequences([prep["text"]])
    padded = pad_sequences(seq, maxlen=300, padding='post')
    prob   = float(bilstm_m1.predict(padded, verbose=0)[0][0])
    pred   = 1 if prob > 0.5 else 0
    return {"class": pred, "prob_real": float(1 - prob), "prob_fake": float(prob),
            "confidence": float(max(prob, 1 - prob) * 100), "prep_info": prep}


def predict_distilbert(text):
    """
    DistilBERT — minimal preprocessing.
    Training used content_raw (basic_clean). DistilBERT's subword tokenizer
    handles punctuation/special chars internally — only lowercase + whitespace.
    Cell 62-66: maxlen=128, padding='max_length', truncation=True
    """
    import torch
    cleaned = text.lower().strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    inputs  = distilbert_tokenizer(
        cleaned, return_tensors="pt",
        padding='max_length', truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    logits = outputs.logits[0].numpy()
    exp    = np.exp(logits - np.max(logits))
    proba  = exp / exp.sum()
    pred   = int(np.argmax(proba))
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100),
            "prep_info": {"original_tokens": len(text.split()),
                          "final_tokens": len(cleaned.split()),
                          "stopwords_removed": 0}}

# ============================================================================
# PDF HELPER — load sample PDFs for download buttons
# ============================================================================
def _load_pdf(filename):
    """Load a PDF file from the same directory as app.py"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path        = os.path.join(current_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("# ⚙️ Configuration")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    ALL_MODELS,
    help="Choose which trained model to use for prediction"
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ Model Information"):
    st.markdown("""
    ### Trained Models Overview
    
    **Label Encoding** (Training Setup):
    - `0` = REAL NEWS ✅
    - `1` = FAKE NEWS 🚨
    
    **ML Models**:
    1. **Logistic Regression** — Test Acc: 85.50%
    2. **SVC** — Test Acc: 85.60% (Best generalization)
    3. **Voting Classifier** — Test Acc: 86.09% (Highest ML)
    
    **DL Models**:
    4. **BiLSTM (with NLP)** — vocab=10000, maxlen=300
    5. **DistilBERT** — Transformer, maxlen=128
    
    **ML Vectorization**:
    - TF-IDF with max 5000 features
    - Trained on 166,418 articles
    - Train/Test split: 80-20
    """)

st.sidebar.markdown("---")

with st.sidebar.expander("📊 Preprocessing Pipeline"):
    st.markdown("""
    ### ML Models (LR, SVC, Voting):
    1. **Lowercase** text
    2. **Remove URLs, Emails**
    3. **Remove Special Chars & Numbers**
    4. **Tokenize** (NLTK)
    5. **Remove Stopwords**
    6. **Lemmatize**
    7. **TF-IDF Vectorize** (5000 features)
    
    ### BiLSTM (with NLP):
    1. **Lowercase** text
    2. **Remove URLs & HTML**
    3. **Keep only letters**
    4. **Remove Stopwords**
    5. **Lemmatize**
    6. **Keras Tokenize** (vocab=10000)
    7. **Pad sequences** (maxlen=300)
    
    ### DistilBERT:
    1. **Lowercase** + whitespace clean
    2. **DistilBERT Tokenizer** (maxlen=128)
    *(Subword tokenizer handles rest)*
    """)

st.sidebar.markdown("---")
st.sidebar.text("📜 Version: 3.0")
st.sidebar.text("👨‍💻 Developed by: Gaurav Upadhyay")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("📰 Fake News Detection System v3.0")
st.markdown("**Comprehensive ML + Deep Learning News Classification**")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "📊 Analytics",
    "📚 Documentation",
    "⚙️ Advanced"
])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.subheader("📝 Analyze News Content")

    col_input1, col_input2 = st.columns([1.5, 1])

    with col_input1:
        title = st.text_input(
            "📌 News Title",
            placeholder="Enter the news headline or title...",
            help="The headline of the news article"
        )
        article = st.text_area(
            "📄 News Article",
            placeholder="Enter the full news content or body...",
            height=200,
            help="The main body of the news article"
        )

    with col_input2:
        st.markdown("### 💡 Quick Actions")

        # ── FAKE NEWS PDF download ──────────────────────────────────────────
        fake_pdf = _load_pdf("fake_news_samples.pdf")
        if fake_pdf:
            st.download_button(
                label="📌 Load Fake Example",
                data=fake_pdf,
                file_name="fake_news_samples.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download 10 fake news examples. Copy any title+text into the fields and predict."
            )
        else:
            st.warning("⚠️ fake_news_samples.pdf not found in project folder")

        # ── REAL NEWS PDF download ──────────────────────────────────────────
        real_pdf = _load_pdf("real_news_samples.pdf")
        if real_pdf:
            st.download_button(
                label="✓ Load Real Example",
                data=real_pdf,
                file_name="real_news_samples.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download 10 real news examples. Copy any title+text into the fields and predict."
            )
        else:
            st.warning("⚠️ real_news_samples.pdf not found in project folder")

        if st.button("🗑️ Clear All", use_container_width=True):
            for k in ["ex_title", "ex_article"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")
        st.info("📥 Download a PDF, pick any news, paste title & text above, then click Analyze")

    # Load session examples if set
    if "ex_title" in st.session_state:
        title = st.session_state["ex_title"]
    if "ex_article" in st.session_state:
        article = st.session_state["ex_article"]

    st.markdown("---")

    if st.button("🚀 Analyze News", use_container_width=True, type="primary"):

        if not title or not article:
            st.error("⚠️ Error: Please enter both title and article content")
            st.stop()

        full_text = f"{title} {article}"

        with st.spinner("🔄 Analyzing news content..."):
            try:
                if selected_model == "Logistic Regression":
                    result = predict_logistic_regression(full_text)
                elif selected_model == "SVC":
                    result = predict_svc(full_text)
                elif selected_model == "Voting Classifier":
                    result = predict_voting_classifier(full_text)
                elif selected_model == "BiLSTM (with NLP)":
                    result = predict_bilstm(full_text)
                else:
                    result = predict_distilbert(full_text)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.stop()

        pred_class = result["class"]
        prob_real  = result["prob_real"]
        prob_fake  = result["prob_fake"]
        confidence = result["confidence"]
        prep_info  = result["prep_info"]

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
        with result_col1:
            if pred_class == 0:
                st.success("### ✅ REAL NEWS")
                st.info("This news appears to be authentic and credible based on the trained model")
            else:
                st.error("### 🚨 FAKE NEWS")
                st.warning("This news appears to contain misinformation or unverified claims")
        with result_col2:
            st.metric("Confidence", f"{confidence:.2f}%")
        with result_col3:
            st.metric("Prediction", "0 = Real\n1 = Fake")

        st.markdown("---")
        st.markdown("### 📈 Probability Distribution")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Real News Probability", f"{prob_real*100:.2f}%", delta_color="normal")
        with metric_col2:
            st.metric("Fake News Probability", f"{prob_fake*100:.2f}%", delta_color="off")

        st.markdown("---")
        st.markdown("### 📊 Probability Visualization")

        fig = go.Figure(data=[
            go.Bar(
                x=['REAL NEWS (0)', 'FAKE NEWS (1)'],
                y=[prob_real, prob_fake],
                marker=dict(
                    color=['#00d400', '#ff4141'],
                    line=dict(color='rgba(255,255,255,0.3)', width=2)
                ),
                text=[f'{prob_real*100:.1f}%', f'{prob_fake*100:.1f}%'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<br>Percentage: %{text}<extra></extra>'
            )
        ])
        fig.update_layout(
            title=f"Prediction Probability Distribution ({selected_model})",
            xaxis_title="News Category (Label)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400,
            template="plotly_dark",
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔧 Text Processing Details")

        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Original Tokens", prep_info["original_tokens"])
        with d2: st.metric("Final Tokens",    prep_info["final_tokens"])
        with d3: st.metric("Removed (Stopwords)", prep_info["stopwords_removed"])
        with d4: st.metric("Text Length", f"{len(full_text)} chars")

        st.markdown("---")
        st.markdown("### ℹ️ Model Information")

        i1, i2, i3 = st.columns(3)
        with i1: st.write(f"**Model**: {selected_model}")
        with i2:
            if selected_model in ["Logistic Regression", "SVC", "Voting Classifier"]:
                st.write("**Vectorizer**: TF-IDF (5000 features)")
            elif selected_model == "BiLSTM (with NLP)":
                st.write("**Vectorizer**: Keras Tokenizer (vocab=10000)")
            else:
                st.write("**Vectorizer**: DistilBERT Tokenizer (maxlen=128)")
        with i3: st.write("**Label Encoding**: 0=Real, 1=Fake")

        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        st.session_state.predictions.append({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title":      title[:40] + "..." if len(title) > 40 else title,
            "model":      selected_model,
            "prediction": "REAL" if pred_class == 0 else "FAKE",
            "confidence": f"{confidence:.2f}%",
            "prob_real":  f"{prob_real*100:.2f}%",
            "prob_fake":  f"{prob_fake*100:.2f}%",
        })

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================
with tab2:
    st.subheader("📊 Prediction History & Statistics")

    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df, use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("### 📈 Statistics")

        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Predictions", len(predictions_df))
        with s2: st.metric("Real News",  len(predictions_df[predictions_df['prediction'] == 'REAL']))
        with s3: st.metric("Fake News",  len(predictions_df[predictions_df['prediction'] == 'FAKE']))
        with s4:
            avg_conf = predictions_df['confidence'].str.rstrip('%').astype(float).mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}%")

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.predictions = []
            st.rerun()
    else:
        st.info("📌 No predictions yet. Go to Prediction tab to analyze news!")

# ============================================================================
# TAB 3: DOCUMENTATION
# ============================================================================
with tab3:
    st.subheader("📚 Complete Documentation")

    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["How to Use", "Preprocessing Pipeline", "Model Details"])

    with doc_tab1:
        st.markdown("""
        ### 📖 How to Use This Application
        
        **Step 1: Enter News Content**
        - Paste the news title in "News Title" field
        - Paste the full article in "News Article" field
        
        **Step 2: Select Model**
        - Choose model from sidebar (ML or DL)
        - Each model has different accuracy/speed tradeoff
        
        **Step 3: Analyze**
        - Click "🚀 Analyze News" button
        - Wait for processing (ML: <1 sec | DL: 1-3 sec)
        
        **Step 4: Review Results**
        - Check prediction (REAL or FAKE)
        - Review confidence percentage
        - Check probability distribution
        
        ### 🎯 Label Encoding
        - **0** = REAL NEWS ✅
        - **1** = FAKE NEWS 🚨
        
        ### 📥 Sample PDFs
        - Click **"📌 Load Fake Example"** to download 10 fake news samples
        - Click **"✓ Load Real Example"** to download 10 real news samples
        - Open the PDF, copy any title + text into the input fields, and click Analyze
        - Fake samples will predict as **FAKE**, Real samples will predict as **REAL**
        
        ### 💡 Pro Tips
        - Longer articles give better predictions
        - Include both title and content
        - Try multiple models for comparison
        - ML models are faster, DL models capture deeper patterns
        """)

    with doc_tab2:
        st.markdown("""
        ### 🔄 Text Preprocessing Pipeline
        
        ---
        #### ML Models (Logistic Regression, SVC, Voting Classifier):
        1. **Lowercase** → "Breaking" → "breaking"
        2. **Remove URLs** → http://, https://, www.
        3. **Remove Emails** → user@domain.com
        4. **Remove Special Chars & Numbers** → !, @, #, 0-9
        5. **Clean Spaces** → Remove extra whitespace
        6. **Tokenize** → NLTK word_tokenize
        7. **Remove Stopwords** → the, and, is, a, an, etc.
        8. **Lemmatize** → breaking→break, controls→control
        9. **TF-IDF Vectorize** → 5000 numerical features
        
        ---
        #### BiLSTM (with NLP):
        1. **Lowercase** → Convert to lowercase
        2. **Remove URLs** → https://, www.
        3. **Remove HTML tags** → <br>, <p>, etc.
        4. **Keep only letters** → remove all non-alphabetic chars
        5. **Remove Stopwords** → NLTK English
        6. **Lemmatize** → WordNetLemmatizer
        7. **Keras Tokenize** → vocab=10000, oov_token="<OOV>"
        8. **Pad sequences** → maxlen=300, padding='post'
        
        ---
        #### DistilBERT:
        1. **Lowercase** → Convert to lowercase
        2. **Clean whitespace** → Remove extra spaces only
        3. **DistilBERT Tokenizer** → maxlen=128, padding='max_length', truncation=True
        *(Subword tokenization handles punctuation/special chars internally)*
        """)

    with doc_tab3:
        st.markdown("""
        ### 🤖 Model Information
        
        **Training Dataset:** 166,418 articles | 50% Real / 50% Fake | 80-20 split
        
        **ML Models:**
        | Model | Train Acc | Test Acc | Gap | Status |
        |-------|-----------|----------|-----|--------|
        | Logistic Regression | 88.86% | 85.50% | 3.36% | Good |
        | SVC | 85.69% | 85.60% | 0.09% | Excellent |
        | Voting Classifier | 88.98% | 86.09% | 2.89% | Good |
        
        **DL Models:**
        
        4. **BiLSTM (with NLP)**
           - Embedding(10000,128) → BiLSTM(64) → Dropout(0.5) → Dense(1, sigmoid)
           - maxlen: 300 | Optimizer: Adam | Loss: Binary Crossentropy
        
        5. **DistilBERT**
           - 6-layer Transformer (distilbert-base-uncased)
           - maxlen: 128 | Fine-tuned for binary classification
        
        ### ✅ Model Selection Tips
        - **Logistic Regression**: Fastest, real-time
        - **SVC**: Best generalization (0.09% overfit gap)
        - **Voting Classifier**: Highest ML accuracy
        - **BiLSTM**: Sequential text patterns
        - **DistilBERT**: Complex language understanding
        """)

# ============================================================================
# TAB 4: ADVANCED
# ============================================================================
with tab4:
    st.subheader("⚙️ Advanced Settings & Information")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.markdown("### 🔧 System Information")
        st.write("**Framework**: Streamlit")
        st.write("**ML Library**: Scikit-learn")
        st.write("**DL Library**: TensorFlow / Keras")
        st.write("**NLP**: NLTK + HuggingFace Transformers")
        st.write("**Vectorizer**: TF-IDF / Keras Tokenizer / DistilBERT Tokenizer")
        st.write("**Version**: 3.0")

    with adv_col2:
        st.markdown("### 📊 Model Statistics")
        st.write("**Total Training Samples**: 166,418")
        st.write("**TF-IDF Features**: 5,000")
        st.write("**BiLSTM vocab**: 10,000 | maxlen: 300")
        st.write("**DistilBERT maxlen**: 128")
        st.write("**Preprocessing Steps (ML)**: 9")
        st.write("**Best ML Accuracy**: 86.09%")
        st.write("**Total Models**: 5")

    st.markdown("---")
    st.markdown("### 📋 Preprocessing Configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **ML Models:**
        - Tokenization: NLTK word_tokenize()
        - Stopwords: English (NLTK)
        - Lemmatization: WordNetLemmatizer
        - Min Token Length: > 2 chars
        
        **BiLSTM:**
        - Tokenization: text.split()
        - Stopwords: English (NLTK)
        - Lemmatization: WordNetLemmatizer
        - Keras Tokenizer: vocab=10000, oov="<OOV>"
        - Padding: post, maxlen=300
        """)
    with c2:
        st.markdown("""
        **ML Cleaning Rules:**
        - Remove URLs: ✅ Yes
        - Remove Emails: ✅ Yes
        - Remove Symbols: ✅ Yes
        - Remove Numbers: ✅ Yes
        - TF-IDF Max Features: 5,000
        - Min DF: 2
        
        **DistilBERT:**
        - Preprocessing: Lowercase + whitespace only
        - Tokenizer: DistilBertTokenizerFast
        - Max Length: 128 | Truncation: ✅
        - Padding: max_length
        """)

    st.markdown("---")
    st.markdown("### ⚠️ Important Notes")
    st.warning("""
    1. **Label Encoding**: 0=Real, 1=Fake (Exact from training)
    2. **Preprocessing**: Exact same pipeline as training (Critical!)
    3. **Accuracy**: ~86% (Not 100% — always verify from multiple sources)
    4. **DL Models**: BiLSTM and DistilBERT may be slower than ML models
    5. **Real-world usage**: This is for educational/demonstration purposes
    6. **Always verify**: Important news should be verified from credible sources
    """)

    st.markdown("---")
    st.markdown("### 🔍 Debug Information")

    db1, db2 = st.columns(2)
    with db1:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir      = os.path.join(current_dir, "V2_Saved_Models_ML")
        dl_dir      = os.path.join(current_dir, "V2_dl_model_saved")
        st.write(f"**Current Directory**: {current_dir}")
        st.write(f"**ML Dir Exists**: {os.path.exists(ml_dir)}")
        st.write(f"**DL Dir Exists**: {os.path.exists(dl_dir)}")
    with db2:
        if os.path.exists(ml_dir):
            st.write("**Files in V2_Saved_Models_ML:**")
            for f in os.listdir(ml_dir):
                st.write(f"  - {f}")
        if os.path.exists(dl_dir):
            st.write("**Files in V2_dl_model_saved:**")
            for f in os.listdir(dl_dir):
                st.write(f"  - {f}")
        else:
            st.error("V2_dl_model_saved folder not found!")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em; padding: 20px;'>
    <p>🎓 Fake News Detection System v3.0</p>
    <p>⚠️ Disclaimer: This tool is for educational purposes. 
    Always verify important news from multiple credible sources.</p>
    <p>📧 Developed with ❤️ by Gaurav Upadhyay</p>
</div>
""", unsafe_allow_html=True)
