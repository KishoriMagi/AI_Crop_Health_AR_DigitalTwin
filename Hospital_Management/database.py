import sqlite3
import os

DB_NAME = 'hospital.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    # Patients Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            address TEXT,
            blood_group TEXT
        )
    ''')

    # Test Requests Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE NOT NULL,
            patient_id TEXT NOT NULL,
            doctor TEXT,
            test_type TEXT,
            sample_type TEXT,
            priority TEXT,
            date TEXT,
            clinical_notes TEXT,
            status TEXT DEFAULT 'Pending',
            result_data TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    ''')

    # Insert default admin user if not exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password_hash, role) VALUES ('admin', 'admin', 'Admin')")

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
