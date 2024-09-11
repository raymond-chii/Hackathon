import datetime
import json
import sqlite3


def setup_database():
    conn = sqlite3.connect("journal_entries.db")
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS entries")

    # Create table for journal entries with updated schema
    cursor.execute(
        """
    CREATE TABLE entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        content TEXT,
        dominant_emotion TEXT,
        color TEXT,
        emotion_breakdown TEXT,
        analysis TEXT
    )
    """
    )

    conn.commit()
    conn.close()


def add_entry(content, mood_data):
    conn = sqlite3.connect("journal_entries.db")
    cursor = conn.cursor()

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
    INSERT INTO entries (date, content, dominant_emotion, color, emotion_breakdown, analysis)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            date,
            content,
            mood_data["dominant_emotion"],
            mood_data["color"],
            json.dumps(mood_data["emotion_breakdown"]),
            mood_data["analysis"],
        ),
    )

    conn.commit()
    conn.close()


def view_entries():
    conn = sqlite3.connect("journal_entries.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT date, content, dominant_emotion, color, emotion_breakdown, analysis FROM entries ORDER BY date DESC"
    )
    entries = cursor.fetchall()

    if not entries:
        print("No entries found.")
    else:
        for (
            date,
            content,
            dominant_emotion,
            color,
            emotion_breakdown,
            analysis,
        ) in entries:
            print(f"\nDate: {date}")
            print(f"Dominant Emotion: {dominant_emotion}")
            print(f"Color: {color}")
            print("Emotion Breakdown:")
            for emotion, percentage in json.loads(emotion_breakdown).items():
                print(f"  {emotion}: {percentage:.2f}%")
            print(f"Analysis: {analysis}")
            print("Content:")
            print(content)
            print("-" * 50)

    conn.close()
