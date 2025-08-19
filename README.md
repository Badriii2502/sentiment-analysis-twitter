# Sentiment Analysis on Tweets (Beginner Project)

A tiny NLP project that classifies short text (tweets) as **positive** or **negative** using TF‑IDF features and a simple Logistic Regression model. Built to be beginner‑friendly and MS‑application ready.

## 🚀 What this contains
- Step‑by‑step Colab instructions (no local setup needed)
- Clean, minimal code using `nltk`'s built‑in **twitter_samples** dataset
- Ready‑to‑upload repo structure + `requirements.txt` and `.gitignore`

## 📦 Dataset
We use NLTK's `twitter_samples` corpus (~10k positive & ~10k negative tweets). No external downloads required once NLTK is installed.

## ✅ Run on Google Colab (recommended)
1. Open **Google Colab** → New Notebook.
2. Run the following cells in order.

**Install & imports**
```python
!pip -q install nltk scikit-learn

import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

**Load dataset**
```python
nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.corpus import twitter_samples

pos = twitter_samples.strings('positive_tweets.json')
neg = twitter_samples.strings('negative_tweets.json')

import pandas as pd
df = pd.DataFrame({'text': pos + neg, 'label': [1]*len(pos) + [0]*len(neg)})
df.head()
```

**Clean text**
```python
import re

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'@\w+', ' ', s)
    s = re.sub(r'#', ' ', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['clean'] = df['text'].apply(clean_text)
```

**Split + vectorize + train**
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(min_df=2, stop_words='english', max_features=10000, ngram_range=(1,2))
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(Xtr, y_train)
```

**Evaluate**
```python
pred = model.predict(Xte)
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred, target_names=['negative','positive']))
```

**Try your own sentences**
```python
def predict_sentiment(text: str):
    c = clean_text(text)
    v = vectorizer.transform([c])
    p = model.predict(v)[0]
    proba = model.predict_proba(v)[0][p]
    label = 'positive' if p == 1 else 'negative'
    return f"{label} (confidence {proba:.2f})"

print(predict_sentiment("I absolutely love this phone!"))
print(predict_sentiment("This is the worst day ever..."))
```

**Save model (optional)**
```python
import joblib
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
```

## 🗂 Suggested repo structure
```
sentiment-analysis-twitter/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ sentiment_twitter_basics.ipynb   # your exported Colab notebook
├─ sentiment_model.joblib           # optional, if you saved it
└─ vectorizer.joblib                # optional
```

## 📈 What accuracy should I expect?
Typically **85–90%** on the held‑out test set. That's great for a starter project.

## 🧩 Ideas to improve
- Try `LinearSVC` and compare accuracy
- Use lemmatization and better text preprocessing
- Switch to IMDB Reviews dataset and compare
- Build a tiny Streamlit/Gradio UI for demo

## 📜 License
MIT — do whatever you want with attribution.
