import colorsys
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


class EmotionClassifier:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, encoding="utf-8")
        self.X = self.df["Answer"]
        self.y = self.df.filter(regex="Answer.f1.*")
        self.y = self.y.apply(pd.to_numeric, errors="coerce")
        self.emotion_labels = self.y.columns

        # Initialize TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(max_features=5000)

        # Initialize Random Forest Classifier
        self.model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.activity_map = {
            "afraid": [
                "Practice deep breathing exercises",
                "Call a friend for support",
                "Write down your fears and challenge them",
            ],
            "angry": [
                "Go for a run or intense exercise",
                "Practice progressive muscle relaxation",
                "Write a letter expressing your feelings but don't send it",
            ],
            "anxious": [
                "Try mindfulness meditation",
                "Make a to-do list to organize your thoughts",
                "Practice yoga",
            ],
            "ashamed": [
                "Write down positive affirmations",
                "Talk to a trusted friend or therapist",
                "Practice self-compassion exercises",
            ],
            "awkward": [
                "Practice social skills through role-playing",
                "Join a social club or group",
                "Read books on improving social confidence",
            ],
            "bored": [
                "Start a new hobby",
                "Learn a new skill online",
                "Reorganize or declutter your space",
            ],
            "calm": [
                "Continue your current relaxation practices",
                "Try a new form of meditation",
                "Engage in a peaceful hobby like gardening",
            ],
            "confused": [
                "Make a pros and cons list",
                "Seek advice from a mentor or expert",
                "Take a break and come back to the problem later",
            ],
            "disgusted": [
                "Practice mindfulness to accept uncomfortable feelings",
                "Engage in a cleansing activity (like cleaning your space)",
                "Express your feelings through art or writing",
            ],
            "excited": [
                "Channel your energy into a creative project",
                "Share your excitement with friends",
                "Plan future goals or events",
            ],
            "frustrated": [
                "Take a short break and practice deep breathing",
                "Break your task into smaller, manageable steps",
                "Talk to someone about the challenges you're facing",
            ],
            "happy": [
                "Express gratitude by thanking someone",
                "Do something kind for others",
                "Engage in your favorite hobby",
            ],
            "jealous": [
                "Practice gratitude for what you have",
                "Set personal goals for self-improvement",
                "Challenge negative thoughts with positive self-talk",
            ],
            "nostalgic": [
                "Look through old photos or mementos",
                "Reconnect with old friends",
                "Start a journal about your favorite memories",
            ],
            "proud": [
                "Celebrate your achievement with loved ones",
                "Reflect on your journey and growth",
                "Set new goals for the future",
            ],
            "sad": [
                "Reach out to a friend for support",
                "Engage in light exercise or a walk in nature",
                "Listen to uplifting music or watch a comforting movie",
            ],
            "satisfied": [
                "Reflect on your accomplishments",
                "Share your satisfaction with others",
                "Set new goals to maintain momentum",
            ],
            "surprised": [
                "Take a moment to process the unexpected event",
                "Share the surprise with others if positive",
                "Reflect on how to adapt to the unexpected",
            ],
        }
        self.color_map = {
            "afraid": (0, 1, 0.5),  # Red
            "angry": (30, 1, 0.5),  # Orange
            "anxious": (60, 1, 0.5),  # Yellow
            "ashamed": (180, 1, 0.3),  # Cyan
            "awkward": (270, 0.5, 0.7),  # Light Purple
            "bored": (0, 0, 0.7),  # Light Gray
            "calm": (180, 1, 0.7),  # Light Blue
            "confused": (270, 1, 0.5),  # Purple
            "disgusted": (120, 1, 0.3),  # Dark Green
            "excited": (45, 1, 0.5),  # Orange-Yellow
            "frustrated": (0, 0.8, 0.4),  # Dark Red
            "happy": (60, 1, 0.7),  # Bright Yellow
            "jealous": (120, 1, 0.5),  # Green
            "nostalgic": (330, 0.5, 0.7),  # Light Pink
            "proud": (300, 1, 0.5),  # Magenta
            "sad": (240, 1, 0.3),  # Dark Blue
            "satisfied": (90, 1, 0.7),  # Light Green
            "surprised": (180, 1, 0.5),  # Cyan
        }
        self.emotion_weights = {
            "happy": 1.0,
            "excited": 0.8,
            "calm": 0.6,
            "satisfied": 0.6,
            "proud": 0.7,
            "surprised": 0.2,
            "bored": -0.2,
            "confused": -0.3,
            "awkward": -0.4,
            "anxious": -0.5,
            "frustrated": -0.6,
            "sad": -0.7,
            "angry": -0.8,
            "afraid": -0.8,
            "disgusted": -0.8,
            "ashamed": -0.9,
            "jealous": -0.9,
            "nostalgic": 0.1,  # Could be positive or negative, so a small positive value
        }

    def calculate_emotional_balance(self, emotions):
        total_score = 0
        total_weight = 0
        for emotion, percentage in emotions:
            if emotion in self.emotion_weights:
                weight = self.emotion_weights[emotion]
                score = weight * (percentage / 100)
                total_score += score
                total_weight += abs(weight) * (percentage / 100)

        if total_weight == 0:
            return 0

        normalized_score = (
            total_score / total_weight + 1
        ) / 2  # Normalize to 0-1 range
        return normalized_score * 100

    def prepare_data(self):
        # Use TF-IDF for feature extraction
        X_tfidf = self.tfidf.fit_transform(self.X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, self.y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

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

        emotion_percentages = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_percentages[emotion] = (
                predictions[i][0][1] * 100
            )  # probability of positive class

        # Clean up emotion labels and filter out low percentages
        cleaned_emotions = [
            (emotion.split(".")[-2], percentage)  # Extract emotion name
            for emotion, percentage in emotion_percentages.items()
            if percentage >= 1.0  # Only include emotions with at least 1% presence
        ]

        # Normalize percentages to sum up to 100
        total_percentage = sum(percentage for _, percentage in cleaned_emotions)
        normalized_emotions = [
            (emotion, (percentage / total_percentage) * 100)
            for emotion, percentage in cleaned_emotions
        ]

        sorted_emotions = sorted(normalized_emotions, key=lambda x: x[1], reverse=True)
        return sorted_emotions

    def suggest_activities(self, emotions, num_suggestions=3):
        top_emotions = emotions[:3]  # Consider top 3 emotions
        suggested_activities = []

        for emotion, percentage in top_emotions:
            if emotion in self.activity_map:
                activities = self.activity_map[emotion]
                suggested_activities.extend(activities)

        # Randomly select num_suggestions activities if we have more than that
        if len(suggested_activities) > num_suggestions:
            suggested_activities = random.sample(suggested_activities, num_suggestions)

        return suggested_activities

    def mix_colors(self, emotions):
        total_h, total_s, total_v, total_weight = 0, 0, 0, 0
        print("Mixing colors for emotions:", emotions)  # Debug print

        for emotion, percentage in emotions:
            if emotion in self.color_map:
                h, s, v = self.color_map[emotion]
                weight = percentage / 100
                total_h += h * weight
                total_s += s * weight
                total_v += v * weight
                total_weight += weight
                print(
                    f"Emotion: {emotion}, Color: {h},{s},{v}, Weight: {weight}"
                )  # Debug print

        if total_weight == 0:
            print("No valid emotions found, returning black")  # Debug print
            return "#000000"  # Default to black if no emotions

        avg_h = total_h / total_weight
        avg_s = total_s / total_weight
        avg_v = total_v / total_weight

        print(f"Average HSV: {avg_h},{avg_s},{avg_v}")  # Debug print

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(avg_h / 360, avg_s, avg_v)

        # Convert RGB to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        print(f"Final color: {hex_color}")  # Debug print
        return hex_color

    def process_entry(self, journal_entry):
        if not journal_entry.strip():
            return None, None, None, None

        emotions = self.predict_emotions(journal_entry)
        if not emotions:
            return None, None, None, None

        color = self.mix_colors(emotions)
        activities = self.suggest_activities(emotions)
        emotional_balance = self.calculate_emotional_balance(emotions)

        return emotions, color, activities, emotional_balance

    def get_past_entries(self, limit=10):
        return self.db.get_entries(limit)

    def close_db(self):
        self.db.close()
