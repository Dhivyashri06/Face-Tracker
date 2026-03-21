import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


class DBHandler:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        self.cur = self.conn.cursor()

    def add_visitor(self):
        now = datetime.now()
        self.cur.execute(
            "INSERT INTO visitors (first_seen, last_seen) VALUES (%s, %s) RETURNING id",
            (now, now)
        )
        visitor_id = self.cur.fetchone()[0]
        self.conn.commit()
        return visitor_id

    def update_visitor_last_seen(self, visitor_id):
        now = datetime.now()
        self.cur.execute(
            "UPDATE visitors SET last_seen=%s WHERE id=%s",
            (now, visitor_id)
        )
        self.conn.commit()

    def log_event(self, visitor_id, event_type, image_path):
        now = datetime.now()
        self.cur.execute(
            "INSERT INTO events (visitor_id, event_type, timestamp, image_path) VALUES (%s, %s, %s, %s)",
            (visitor_id, event_type, now, image_path)
        )
        self.conn.commit()

    def get_unique_visitor_count(self):
        self.cur.execute("SELECT COUNT(*) FROM visitors")
        return self.cur.fetchone()[0]

    def close(self):
        self.cur.close()
        self.conn.close()
