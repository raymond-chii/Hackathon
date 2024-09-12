import sqlite3
from datetime import datetime

class JournalDatabase:
    def __init__(self, db_name='journal.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY,
            date TEXT,
            entry TEXT,
            color TEXT
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            emotion TEXT,
            percentage REAL,
            FOREIGN KEY (entry_id) REFERENCES entries (id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            activity TEXT,
            FOREIGN KEY (entry_id) REFERENCES entries (id)
        )
        ''')
        self.conn.commit()

    def add_entry(self, entry, emotions, color, activities):
        cursor = self.conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO entries (date, entry, color) VALUES (?, ?, ?)",
                       (date, entry, color))
        entry_id = cursor.lastrowid

        for emotion, percentage in emotions:
            cursor.execute("INSERT INTO emotions (entry_id, emotion, percentage) VALUES (?, ?, ?)",
                           (entry_id, emotion, percentage))

        for activity in activities:
            cursor.execute("INSERT INTO activities (entry_id, activity) VALUES (?, ?)",
                           (entry_id, activity))

        self.conn.commit()

    def get_entries(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT e.id, e.date, e.entry, e.color, 
               GROUP_CONCAT(DISTINCT em.emotion || ':' || em.percentage),
               GROUP_CONCAT(DISTINCT a.activity)
        FROM entries e
        LEFT JOIN emotions em ON e.id = em.entry_id
        LEFT JOIN activities a ON e.id = a.entry_id
        GROUP BY e.id
        ORDER BY e.date DESC
        LIMIT ?
        """, (limit,))
        return cursor.fetchall()

    def close(self):
        self.conn.close()