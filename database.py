"""
database.py
-----------
Handles all SQLite database operations for the Smart Attendance System.

Tables:
    students   - registered students
    attendance - one row per student per day (UNIQUE on student_id + date)

Other files should call the functions here instead of writing SQL themselves.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

# The database file is created next to this script, whatever folder we run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


# ----------------------------------------------------------------------
# Connection helpers
# ----------------------------------------------------------------------
def get_connection():
    """Open a connection to attendance.db. Rows can be read like dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session():
    """
    Open a connection, save (commit) changes if everything succeeds,
    undo (rollback) if an error happens, and ALWAYS close the connection.

    Usage:
        with db_session() as conn:
            conn.execute("...")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Create tables
# ----------------------------------------------------------------------
def create_database():
    """Create attendance.db and its two tables if they do not exist yet."""
    try:
        with db_session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id        TEXT NOT NULL UNIQUE,
                    name              TEXT NOT NULL,
                    branch            TEXT,
                    semester          INTEGER,
                    email             TEXT,
                    registration_date TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    date       TEXT NOT NULL,
                    time       TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'Present',
                    UNIQUE (student_id, date)
                )
                """
            )
        return True
    except sqlite3.Error as e:
        print(f"[DB ERROR] Could not create database: {e}")
        return False


# ----------------------------------------------------------------------
# Student functions
# ----------------------------------------------------------------------
def add_student(student_id, name, branch="", semester=None, email=""):
    """
    Register a new student.
    Returns (success, message).
    """
    student_id = str(student_id).strip()
    name = str(name).strip()
    branch = str(branch).strip()
    email = str(email).strip()

    if not student_id:
        return False, "Student ID cannot be empty."
    if not name:
        return False, "Student name cannot be empty."

    registration_date = datetime.now().strftime("%Y-%m-%d")

    try:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO students
                    (student_id, name, branch, semester, email, registration_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (student_id, name, branch, semester, email, registration_date),
            )
        return True, f"Student {name} (ID {student_id}) registered successfully."
    except sqlite3.IntegrityError:
        # Raised because student_id is UNIQUE and already exists
        return False, f"Student ID {student_id} already exists."
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def get_student(student_id):
    """Return one student as a dictionary, or None if not found."""
    student_id = str(student_id).strip()
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"[DB ERROR] get_student failed: {e}")
        return None


def get_all_students():
    """Return a list of all students (each one a dictionary)."""
    try:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM students ORDER BY CAST(student_id AS INTEGER), student_id"
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"[DB ERROR] get_all_students failed: {e}")
        return []


def student_exists(student_id):
    """Return True if a student with this ID is registered."""
    student_id = str(student_id).strip()
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()
        return row is not None
    except sqlite3.Error as e:
        print(f"[DB ERROR] student_exists failed: {e}")
        return False


def delete_student(student_id):
    """
    Delete a student AND that student's attendance records.
    (Deleting the face images in dataset/ is handled elsewhere.)
    Returns (success, message).
    """
    student_id = str(student_id).strip()
    try:
        with db_session() as conn:
            cursor = conn.execute(
                "DELETE FROM students WHERE student_id = ?", (student_id,)
            )
            if cursor.rowcount == 0:
                return False, f"No student found with ID {student_id}."
            conn.execute(
                "DELETE FROM attendance WHERE student_id = ?", (student_id,)
            )
        return True, f"Student {student_id} and their attendance records were deleted."
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


# ----------------------------------------------------------------------
# Attendance functions
# ----------------------------------------------------------------------
def attendance_exists(student_id, date_str=None):
    """
    Has this student already been marked for the given date?
    If date_str is None, today's date is used.
    """
    student_id = str(student_id).strip()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
                (student_id, date_str),
            ).fetchone()
        return row is not None
    except sqlite3.Error as e:
        print(f"[DB ERROR] attendance_exists failed: {e}")
        return False


def mark_attendance(student_id):
    """
    Mark a student PRESENT for today.

    Returns (status, message) where status is one of:
        "marked"          - a new attendance row was saved
        "already_marked"  - the student already has a record for today
        "not_found"       - the student ID is not registered
        "error"           - a database error happened
    """
    student_id = str(student_id).strip()
    student = get_student(student_id)
    if student is None:
        return "not_found", f"Student ID {student_id} is not registered."

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO attendance (student_id, name, date, time, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, student["name"], date_str, time_str, "Present"),
            )
        return "marked", f"{student['name']} marked Present at {time_str}."
    except sqlite3.IntegrityError:
        # The UNIQUE (student_id, date) rule stopped a duplicate
        return "already_marked", f"{student['name']} is already marked Present today."
    except sqlite3.Error as e:
        return "error", f"Database error: {e}"


def get_attendance(date_str=None, student_id=None, name=None, branch=None):
    """
    Return attendance records as a list of dictionaries.
    Every filter is optional; give none to get everything.
        date_str   : exact date, "YYYY-MM-DD"
        student_id : exact student ID
        name       : part of a name (case-insensitive search)
        branch     : exact branch, e.g. "CSE"
    """
    query = """
        SELECT a.student_id, a.name, s.branch, a.date, a.time, a.status
        FROM attendance a
        LEFT JOIN students s ON a.student_id = s.student_id
    """
    conditions = []
    params = []

    if date_str:
        conditions.append("a.date = ?")
        params.append(date_str)
    if student_id:
        conditions.append("a.student_id = ?")
        params.append(str(student_id).strip())
    if name:
        conditions.append("a.name LIKE ?")
        params.append(f"%{str(name).strip()}%")
    if branch:
        conditions.append("s.branch = ?")
        params.append(str(branch).strip())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.date DESC, a.time DESC"

    try:
        with db_session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"[DB ERROR] get_attendance failed: {e}")
        return []


def get_today_attendance():
    """Return today's attendance records."""
    today = datetime.now().strftime("%Y-%m-%d")
    return get_attendance(date_str=today)


# ----------------------------------------------------------------------
# Attendance percentage
# ----------------------------------------------------------------------
def get_total_working_days():
    """
    Total working days = number of DIFFERENT dates that have at least
    one attendance record. (A day on which attendance was taken.)
    """
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT date) AS days FROM attendance"
            ).fetchone()
        return row["days"]
    except sqlite3.Error as e:
        print(f"[DB ERROR] get_total_working_days failed: {e}")
        return 0


def get_present_days(student_id):
    """Number of days this student was marked Present."""
    student_id = str(student_id).strip()
    try:
        with db_session() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS days FROM attendance
                WHERE student_id = ? AND status = 'Present'
                """,
                (student_id,),
            ).fetchone()
        return row["days"]
    except sqlite3.Error as e:
        print(f"[DB ERROR] get_present_days failed: {e}")
        return 0


def calculate_attendance_percentage(student_id):
    """
    Attendance % = (Present Days / Total Working Days) x 100
    Returns 0.0 if no attendance has been taken yet.
    """
    total_days = get_total_working_days()
    if total_days == 0:
        return 0.0
    present_days = get_present_days(student_id)
    return round((present_days / total_days) * 100, 2)


# ----------------------------------------------------------------------
# Run this file directly to create the database:  python database.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if create_database():
        print("Database created successfully.")
        print(f"Database file: {DB_PATH}")
        try:
            with db_session() as conn:
                rows = conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
            print("Tables found: " + ", ".join(row["name"] for row in rows))
        except sqlite3.Error as e:
            print(f"[DB ERROR] Could not list tables: {e}")
    else:
        print("Database creation FAILED. See the error message above.")