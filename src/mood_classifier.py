import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


class EmotionClassifier:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, encoding="utf-8")
        self.X = self.df["Answer"]
        self.y = self.df.filter(regex="Answer.f1.*")
        self.y = self.y.apply(pd.to_numeric, errors="coerce")
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.emotion_labels = self.y.columns

    def prepare_data(self):
        X_tfidf = self.tfidf.fit_transform(self.X)
        return train_test_split(X_tfidf, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return classification_report(
            y_test, y_pred, target_names=self.emotion_labels, zero_division=0
        )

    def predict_emotions(self, journal_entry):
        entry_tfidf = self.tfidf.transform([journal_entry])
        predictions = self.model.predict_proba(entry_tfidf)
        percentages = predictions[0] * 100

        total = np.sum(percentages)
        normalized_percentages = (percentages / total) * 100

        emotion_percentages = dict(zip(self.emotion_labels, normalized_percentages))
        sorted_emotions = sorted(
            emotion_percentages.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_emotions
