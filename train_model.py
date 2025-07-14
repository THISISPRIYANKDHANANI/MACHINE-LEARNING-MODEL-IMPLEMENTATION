# ✅ train_and_save_model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("spam_updated.csv")
df = df[['label', 'message']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 2: Preprocessing
X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_vectorized = vectorizer.fit_transform(X)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate
# y_pred = model.predict(X_test)
# print("✅ Accuracy:", accuracy_score(y_test, y_pred))
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# plt.figure(figsize=(5, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# Step 6: Save the model and vectorizer
# if not os.path.exists("model"):
#     os.makedirs("model")

# joblib.dump(vectorizer, "model/vectorizer.pkl")
# joblib.dump(model, "model/spam_model.pkl")
# print("✅ Model and vectorizer saved successfully!")

# Step 7: Optional Test
def predict_message(msg):
    vec = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]
    return "📩 Spam" if prediction == 1 else "✅ Ham (Not Spam)"

# # Test example
# test_msg = """Congratulations Uday!

# Your email ID has been selected in our monthly draw 🎉
# You are now eligible to claim a *Brand New iPhone 15* — absolutely FREE!

# This is a limited-time offer, and we only have 3 units left!
# Claim your reward before the offer expires tonight at 11:59 PM.

# 👉 Confirm your shipping address now and get it delivered at your doorstep.

# Thank you for participating,
# Rewards Team
# GlobalGiveaways Inc.
# """
# print("\n🔍 Prediction for test message:\n", test_msg)
# print("Result:", predict_message(test_msg))

# Expose model and vectorizer for import
trained_vectorizer = vectorizer
trained_model = model