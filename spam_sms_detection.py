# spam_sms_detection.py
# Advanced Spam SMS Detection Project
# Run:
# pip install pandas scikit-learn nltk joblib numpy
# python spam_sms_detection.py

import pandas as pd
import numpy as np
import re
import joblib
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# Download NLTK data safely
# ---------------------------------------------------
def setup_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download("stopwords")
        nltk.download("wordnet")

setup_nltk()

# ---------------------------------------------------
# Main Class
# ---------------------------------------------------
class AdvancedSpamDetector:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # ML Pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=self.clean_text,
                ngram_range=(1, 2),
                max_features=5000
            )),
            ("classifier", MultinomialNB())
        ])

        # Parameter Grid
        self.param_grid = {
            "tfidf__max_df": [0.8, 1.0],
            "classifier__alpha": [0.1, 0.5, 1.0]
        }

    # ---------------------------------------------------
    # Clean Text Function
    # ---------------------------------------------------
    def clean_text(self, text):

        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)

        words = []

        for word in text.split():
            if word not in self.stop_words:
                word = self.lemmatizer.lemmatize(word)
                words.append(word)

        return " ".join(words)

    # ---------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------
    def load_dataset(self):

        print("Loading Dataset...")

        data = {
            "label": [
                "ham", "spam", "ham", "spam", "ham",
                "spam", "ham", "spam", "ham", "spam",
                "ham", "spam", "ham", "spam", "ham",
                "spam", "ham", "spam", "ham", "spam"
            ],

            "message": [
                "Hey how are you",
                "Win free cash prize now",
                "Call me later",
                "Claim free recharge now",
                "Meet me tomorrow",
                "Limited offer click now",
                "Where are you",
                "Congratulations you won lottery",
                "See you soon",
                "Win iphone now",
                "Let's go market",
                "Urgent claim reward",
                "Come home early",
                "Free membership available",
                "How was your day",
                "Get money instantly",
                "Can we talk later",
                "Buy now offer ends today",
                "Good morning friend",
                "Exclusive reward waiting"
            ]
        }

        df = pd.DataFrame(data)

        df["label_num"] = df["label"].map({
            "ham": 0,
            "spam": 1
        })

        return df

    # ---------------------------------------------------
    # Train Model
    # ---------------------------------------------------
    def train_model(self, df):

        print("Training Model...")

        X = df["message"]
        y = df["label_num"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        grid = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=3,
            scoring="accuracy"
        )

        grid.fit(self.X_train, self.y_train)

        self.pipeline = grid.best_estimator_

        print("Best Parameters:", grid.best_params_)

    # ---------------------------------------------------
    # Evaluate Model
    # ---------------------------------------------------
    def evaluate_model(self):

        pred = self.pipeline.predict(self.X_test)

        print("\n========== RESULT ==========")
        print("Accuracy:", round(accuracy_score(self.y_test, pred) * 100, 2), "%")

        print("\nClassification Report:\n")
        print(classification_report(
            self.y_test,
            pred,
            target_names=["Ham", "Spam"]
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, pred))
        print("============================\n")

    # ---------------------------------------------------
    # Save Model
    # ---------------------------------------------------
    def save_model(self):

        joblib.dump(self.pipeline, "spam_model.pkl")
        print("Model Saved as spam_model.pkl")

    # ---------------------------------------------------
    # Predict Message
    # ---------------------------------------------------
    def predict_message(self, msg):

        prediction = self.pipeline.predict([msg])[0]

        confidence = np.max(
            self.pipeline.predict_proba([msg])
        ) * 100

        if prediction == 1:
            return f"SPAM 🚨 ({confidence:.2f}% confidence)"
        else:
            return f"HAM ✅ ({confidence:.2f}% confidence)"

# ---------------------------------------------------
# Main Program
# ---------------------------------------------------
if __name__ == "__main__":

    print("🚀 Advanced Spam SMS Detector Started\n")

    detector = AdvancedSpamDetector()

    # Load Dataset
    df = detector.load_dataset()

    # Train Model
    detector.train_model(df)

    # Evaluate
    detector.evaluate_model()

    # Save
    detector.save_model()

    # User Testing
    print("========== LIVE TEST ==========")

    while True:

        msg = input("Enter SMS Message (type exit): ")

        if msg.lower() == "exit":
            print("Program Closed.")
            break

        elif msg.strip() == "":
            print("Please enter valid message.\n")

        else:
            result = detector.predict_message(msg)
            print("Prediction:", result)
            print()