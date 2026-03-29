import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. SETUP
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 2. LOAD DATA
df = pd.read_csv('3) Sentiment dataset.csv')
df = df.dropna(subset=['Text', 'Sentiment'])

# --- THE "LEVEL 3" GROUPING FIX ---
# This converts 100+ emotions into 3 simple categories for a cleaner model
def map_sentiment(label):
    label = str(label).strip().lower()
    pos = ['acceptance', 'admiration', 'affection', 'amusement', 'anticipation', 'awe', 'joy', 'love', 'spark', 'thrill', 'vibrancy', 'pride', 'optimism', 'elation', 'delight', 'happiness', 'pleasure']
    neg = ['anger', 'annoyance', 'disgust', 'fear', 'sadness', 'sorrow', 'remorse', 'grief', 'hate', 'bitterness', 'disappointment']
    
    if label in pos: return 'Positive'
    if label in neg: return 'Negative'
    return 'Neutral'

df['Sentiment_Group'] = df['Sentiment'].apply(map_sentiment)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(clean_tokens)

df['Clean_Text'] = df['Text'].apply(clean_text)

# 3. VECTORIZE
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['Clean_Text']).toarray()
y = df['Sentiment_Group']

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. EVALUATION
y_pred = model.predict(X_test)

print("\n--- Level 3: NLP Task 2 Results (Categorized) ---")
print(f"New Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. VISUALIZE
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Consolidated Sentiment Analysis')
plt.show()