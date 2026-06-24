import numpy as np
import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load saved components
model = joblib.load('best_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.title("🛡️ Fake News Detector (BI Edition)")
st.subheader("Analyze news authenticity using Machine Learning")

user_input = st.text_area("Paste the news article text here:", height=200)

if st.button("Analyze Authenticity"):
    if user_input:
        # Preprocess and Predict
        # (Assuming the same clean_text function is imported)
        cleaned = clean_text(user_input)
        data = tfidf.transform([cleaned])
        prediction = model.predict(data)
        probability = model.predict_proba(data)
        
        result = "FAKE" if prediction[0] == 1 else "REAL"
        confidence = np.max(probability) * 100
        
        if result == "FAKE":
            st.error(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
            st.info("💡 BI Insight: This text matches linguistic patterns found in misinformation datasets.")
        else:
            st.success(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
    else:
        st.warning("Please enter some text to analyze.")
st.markdown("---")
st.caption("Detecting misinformation using AI-driven text classification, ML, and NLP.")