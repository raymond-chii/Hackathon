import sqlite3
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Entry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    entry = db.Column(db.Text)
    color = db.Column(db.String(7))
    emotional_balance = db.Column(db.Float)

    emotions = db.relationship("Emotion", backref="entry", lazy=True)
    activities = db.relationship("Activity", backref="entry", lazy=True)


class Emotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("entry.id"))
    emotion = db.Column(db.String(50))
    percentage = db.Column(db.Float)


class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("entry.id"))
    activity = db.Column(db.String(200))


def create_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS entry (
        id INTEGER PRIMARY KEY,
        date DATETIME,
        entry TEXT,
        color VARCHAR(7),
        emotional_balance FLOAT
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS emotion (
        id INTEGER PRIMARY KEY,
        entry_id INTEGER,
        emotion VARCHAR(50),
        percentage FLOAT,
        FOREIGN KEY (entry_id) REFERENCES entry (id)
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS activity (
        id INTEGER PRIMARY KEY,
        entry_id INTEGER,
        activity VARCHAR(200),
        FOREIGN KEY (entry_id) REFERENCES entry (id)
    )
    """
    )

    conn.commit()
    conn.close()


def check_and_add_column(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the column exists
    cursor.execute("PRAGMA table_info(entry)")
    columns = [column[1] for column in cursor.fetchall()]

    if "emotional_balance" not in columns:
        print("Adding emotional_balance column to entry table")
        cursor.execute("ALTER TABLE entry ADD COLUMN emotional_balance FLOAT")

    conn.commit()
    conn.close()


def add_entry(entry_text, emotions, color, activities, emotional_balance):
    new_entry = Entry(
        entry=entry_text, color=color, emotional_balance=emotional_balance
    )
    db.session.add(new_entry)
    db.session.flush()

    for emotion, percentage in emotions:
        new_emotion = Emotion(
            entry_id=new_entry.id, emotion=emotion, percentage=percentage
        )
        db.session.add(new_emotion)

    for activity in activities:
        new_activity = Activity(entry_id=new_entry.id, activity=activity)
        db.session.add(new_activity)

    db.session.commit()


def get_entries(limit=10):
    entries = Entry.query.order_by(Entry.date.desc()).limit(limit).all()
    result = []
    for entry in entries:
        emotions = Emotion.query.filter_by(entry_id=entry.id).all()
        activities = Activity.query.filter_by(entry_id=entry.id).all()
        result.append(
            {
                "id": entry.id,
                "date": entry.date,
                "entry": entry.entry,
                "color": entry.color,
                "emotional_balance": entry.emotional_balance,
                "emotions": [(e.emotion, e.percentage) for e in emotions],
                "activities": [a.activity for a in activities],
            }
        )
    return result


def get_emotional_balance_history(days=30):
    entries = Entry.query.order_by(Entry.date.desc()).limit(days).all()
    return [
        (entry.date.isoformat(), entry.emotional_balance) for entry in reversed(entries)
    ]


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database tables created.")
