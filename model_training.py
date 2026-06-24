import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# --- Phase 1: Data Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in brackets
    text = re.sub(r'\\W', " ", text) # Remove special chars
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Punctuation
    text = re.sub(r'\n', '', text) # New lines
    text = re.sub(r'\w*\d\w*', '', text) # Words with numbers
    
    # Tokenization & Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- Phase 2: Load and Prepare Data ---
def load_and_preprocess():
    # Assuming files are in current directory
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    
    fake['label'] = 1 # Fake
    true['label'] = 0 # Real
    
    df = pd.concat([fake, true], axis=0).reset_index(drop=True)
    df['content'] = df['title'] + " " + df['text']
    df['content'] = df['content'].apply(clean_text)
    return df

# --- Phase 3: Model Building ---
def train_models(df):
    X = df['content']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "report": classification_report(y_test, pred)
        }
    
    # Save the best model (Example: Logistic Regression) and Vectorizer
    joblib.dump(models["Naive Bayes"], 'best_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    
    return results, y_test, pred
if __name__ == "__main__":
    df = load_and_preprocess()
    results, y_test, pred = train_models(df)

    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {result['accuracy']}")
        print(result['report'])