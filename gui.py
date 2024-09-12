import matplotlib

matplotlib.use("Agg")  # Use the non-GUI backend for rendering plots

import base64
import io
import os

import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request

from db.database import (
    Activity,
    Emotion,
    Entry,
    add_entry,
    db,
    get_emotional_balance_history,
    get_entries,
    init_db,
)
from mood_classifier import EmotionClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, "templates")

app = Flask(__name__, template_folder=template_dir)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///journal.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database
init_db(app)

classifier = EmotionClassifier("data.csv")
classifier.train_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    entry = request.form["entry"]
    emotions, color, activities, emotional_balance = classifier.process_entry(entry)

    if emotions is None:
        return jsonify(
            {
                "error": "No significant emotions detected. Please try a more detailed entry."
            }
        )

    # Add entry to the database
    add_entry(entry, emotions, color, activities, emotional_balance)

    # Create pie chart for emotions
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = [emotion for emotion, _ in emotions]
    sizes = [percentage for _, percentage in emotions]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")

    # Convert plot to base64 image for embedding
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return jsonify(
        {
            "emotions": emotions,
            "color": color,
            "activities": activities,
            "emotional_balance": emotional_balance,
            "plot": plot_data,
        }
    )


@app.route("/emotional_balance_history")
def emotional_balance_history():
    history = get_emotional_balance_history()
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)
