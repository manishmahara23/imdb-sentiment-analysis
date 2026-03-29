import streamlit as st
import torch
import pickle
import re
import numpy as np
import config
from sample_review import EXAMPLE_REVIEWS
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

# ---------------- CONFIG ----------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
MAX_LEN = config.MAX_LEN
BEST_THRESHOLD = config.BEST_THRESHOLD  # or your ROC-optimized threshold

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- MODEL DEFINITION ----------------
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size = embed_dim,
            hidden_size = hidden_dim,
            dropout = 0.3,
            batch_first=True,
            bidirectional=True,
            num_layers = 2
        )
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.fc_dropout(combined)

        return self.sigmoid(self.fc(combined)).squeeze()
# ---------------- LOAD MODEL & VOCAB ----------------
@st.cache_resource
def load_model():
    with open(MODEL_DIR/'vocab.pkl', "rb") as f:
        word2idx = pickle.load(f)

    model = BiLSTMModel(
        len(word2idx),
        embed_dim = config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM
        )
    model.load_state_dict(torch.load(MODEL_DIR/'model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, word2idx

model, word2idx = load_model()

# ---------------- PREPROCESS INPUT ----------------
def preprocess(text):
    text = clean_text(text)
    tokens = text.split()

    encoded = [
        word2idx.get(word, word2idx["<OOV>"])
        for word in tokens
    ]

    if len(encoded) >= MAX_LEN:
        encoded = encoded[:MAX_LEN]
    else:
        encoded += [word2idx["<PAD>"]] * (MAX_LEN - len(encoded))

    return torch.tensor(encoded).unsqueeze(0).to(DEVICE)

# ---------------- UI ----------------
st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="centered")

st.title("🎬 IMDb Sentiment Analyzer")
st.write("Enter a movie review to predict its sentiment.")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "Movie Review",
    height=200,
    value=st.session_state.user_input,
    placeholder="Type or paste a movie review here..."
)

st.markdown("### ✨ Try an Example Review")



example_items = list(EXAMPLE_REVIEWS.items())

row1 = example_items[:4]   
row2 = example_items[4:]


cols1 = st.columns(len(row1))
for col, (label, text) in zip(cols1, row1):
    if col.button(label):
        st.session_state.user_input = text
        st.rerun()

cols2 = st.columns(len(row2))
for col, (label, text) in zip(cols2, row2):
    if col.button(label):
        st.session_state.user_input = text
        st.rerun()

if st.button("Analyze Sentiment", type = 'primary'):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_tensor = preprocess(user_input)
        with torch.no_grad():
            prob = model(input_tensor).item()

        sentiment = "Positive 😊👍🏽" if prob >= BEST_THRESHOLD else "Negative 😞👎🏽"

        st.subheader("Prediction")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {prob:.4f}")

