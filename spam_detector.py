
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
import nltk

# Download stopwords (only first time)
nltk.download('stopwords')

# Load Dataset (Ensure 'spam.csv' is in the same folder)
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only needed columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Show first few rows
print("Sample Data:\n", data.head(), "\n")

# Clean text: remove punctuation & lowercase
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([ch for ch in msg if ch not in string.punctuation])
    return msg

data['message'] = data['message'].apply(clean_text)

# Feature Extraction: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with custom messages
test_msgs = [
    "Congratulations! You have won a free ticket. Reply YES to claim.",
    "Hey, are we still meeting for lunch today?"
]

test_vectors = vectorizer.transform(test_msgs)
preds = model.predict(test_vectors)

for text, label in zip(test_msgs, preds):
    print(f"\nMessage: {text}\nPrediction: {label}")