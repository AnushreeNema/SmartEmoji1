import streamlit as st
import torch
import math
import torch.nn.functional as F
import pickle
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
st.set_page_config(page_title="SmartEmoji", layout="centered")
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "bertweet_model_final", "bertweet_model")
    encoder_path = os.path.join(BASE_DIR, "bertweet_model_final", "label_encoder.pkl")
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bertweet-base",
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    return tokenizer, model, le
tokenizer, model, le = load_model()
emotion_to_emoji = {
    "sob": "😭",          
    "heart_eyes": "😍",   
    "weary": "😩",        
    "blush": "😊",        
    "wink": "😉",         
    "yum": "😋",          
    "smirk": "😏",        
    "grin": "😁",         
    "relaxed": "😌",      
    "flushed": "😳"      
}
emoji_counts = {
    "sob": 50525,
    "heart_eyes": 39193,
    "weary": 26855,
    "blush": 22894,
    "wink": 18078,
    "yum": 16790,
    "smirk": 15231,
    "grin": 15138,
    "relaxed": 10472,
    "flushed": 10155
}
total = sum(emoji_counts.values())
emoji_freq = {k: v / total for k, v in emoji_counts.items()}
emoji_penalty = {}
for k, v in emoji_freq.items():
    emoji_penalty[k] = 1 / math.sqrt(v)
def sentiment_score(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"
def sentiment_alignment(emotion, sentiment):
    positive_emotions = ["joy", "heart_eyes", "grin", "wink", "relaxed"]
    negative_emotions = ["sob", "weary"]

    if sentiment == "positive" and emotion in positive_emotions:
        return 1.0
    elif sentiment == "negative" and emotion in negative_emotions:
        return 1.0
    elif sentiment == "neutral":
        return 0.6
    else:
        return 0.2
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
    sentiment = sentiment_score(text)
    results = []
    for i, prob in enumerate(probs):
        label = le.inverse_transform([i])[0]
        align = sentiment_alignment(label, sentiment)
        freq_penalty = emoji_penalty.get(label, 1.0)
        freq_penalty = max(0.7, freq_penalty)
        # prevent over-penalizing
        freq_penalty = max(0.7, freq_penalty)
        final_score = prob * align * freq_penalty
        results.append({
            "label": label,
            "prob": float(prob),
            "alignment": align,
            "final_score": final_score
        })
    # sort by final score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:3], sentiment
st.markdown("<h1 style='text-align:center;'>SmartEmoji</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Find the best emoji for your tweets</p>", unsafe_allow_html=True)
st.markdown("### Enter your tweet:")
user_input = st.text_area("", placeholder="e.g. Just finished my project after a sleepless nights")
if st.button("Suggest Emojis"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
        st.stop()
    preds, sentiment = predict(user_input)
    st.markdown(f"### Detected sentiment: **{sentiment.upper()}**")
    st.markdown("##Recommended Emojis")
    cols = st.columns(len(preds))
    for i, item in enumerate(preds):
        emoji_char = emotion_to_emoji.get(item["label"], "🙂")
        with cols[i]:
            st.markdown(f"<h1 style='text-align:center'>{emoji_char}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center'>{item['label']}</p>", unsafe_allow_html=True)
            normalized = item["final_score"] / max([p["final_score"] for p in preds])
            st.progress(float(normalized))
    st.markdown("## Why these emojis?")
    for item in preds:
        st.markdown(
            f"""
            **{item['label']}**
            - Model confidence: {item['prob']:.2f}
            - Sentiment match: {item['alignment']}
            - Final score: **{item['final_score']:.2f}**
            """
        )
    st.markdown("##Final Ranking Scores")
    df = pd.DataFrame(preds)
    df = df.sort_values(by="final_score", ascending=False)
    st.bar_chart(df.set_index("label")["final_score"])