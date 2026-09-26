"""Amazon Fine Food Review Sentiment Analyzer - Streamlit demo."""
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="🍽️", layout="centered")

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("rnn_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

st.title("🍽️ Review Sentiment Analyzer")
st.caption("An LSTM (Recurrent Neural Network) trained on 110,000+ Amazon Fine Food reviews. 94.3% accuracy on held-out test data.")

text = st.text_area("Paste a product review:", height=140,
                     placeholder="e.g. This coffee is fantastic, rich flavor and great aroma every morning.")

if st.button("Analyze Sentiment", type="primary", disabled=not text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded, verbose=0)[0]
    neg, pos = pred[0], pred[1]

    if pos > neg:
        st.success(f"**Positive** ({pos*100:.1f}% confidence)")
    else:
        st.error(f"**Negative** ({neg*100:.1f}% confidence)")

    st.progress(float(pos), text=f"Positive: {pos*100:.1f}%")
    st.progress(float(neg), text=f"Negative: {neg*100:.1f}%")

st.divider()
st.caption("Model: LSTM (Long Short-Term Memory) · Trained on Amazon Fine Food Reviews dataset · Built by Kuppe Labs")
