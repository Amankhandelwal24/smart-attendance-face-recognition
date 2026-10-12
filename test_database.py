"""
test_database.py
----------------
Tests the functions in database.py using a TEMPORARY database
(test_attendance.db). Your real attendance.db is NOT touched.

Run:  python test_database.py
"""

import os

import database

# Point database.py at a temporary test database instead of the real one.
TEST_DB = os.path.join(database.BASE_DIR, "test_attendance.db")
database.DB_PATH = TEST_DB

passed = 0
failed = 0


def check(description, condition):
    """Print PASS or FAIL for one check and keep count."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {description}")
    else:
        failed += 1
        print(f"  FAIL: {description}")


def run_tests():
    print("\n--- Creating test database ---")
    check("create_database() returns True", database.create_database() is True)

    print("\n--- Student registration ---")
    ok, msg = database.add_student("101", "Rahul Sharma", "CSE", 8, "rahul@example.com")
    check("add_student 101 succeeds", ok)

    ok, msg = database.add_student("101", "Someone Else", "CSE", 8, "x@example.com")
    check("duplicate student ID 101 is rejected", not ok)
    print(f"        message: {msg}")

    ok, msg = database.add_student("", "No ID", "CSE", 8, "x@example.com")
    check("empty student ID is rejected", not ok)

    ok, msg = database.add_student("150", "   ", "CSE", 8, "x@example.com")
    check("empty student name is rejected", not ok)

    check("student_exists('101') is True", database.student_exists("101") is True)
    check("student_exists('999') is False", database.student_exists("999") is False)

    student = database.get_student("101")
    check("get_student('101') returns Rahul Sharma",
          student is not None and student["name"] == "Rahul Sharma")
    check("get_student('999') returns None", database.get_student("999") is None)

    print("\n--- Marking attendance ---")
    status, msg = database.mark_attendance("101")
    check("first mark_attendance('101') -> 'marked'", status == "marked")
    print(f"        message: {msg}")

    status, msg = database.mark_attendance("101")
    check("second mark_attendance('101') same day -> 'already_marked'",
          status == "already_marked")
    print(f"        message: {msg}")

    check("attendance_exists('101') is True", database.attendance_exists("101") is True)
    check("attendance_exists('102') is False", database.attendance_exists("102") is False)

    status, msg = database.mark_attendance("999")
    check("mark_attendance for unregistered ID -> 'not_found'", status == "not_found")

    today_records = database.get_today_attendance()
    check("only ONE attendance row exists today (no duplicate)", len(today_records) == 1)

    print("\n--- Attendance percentage ---")
    ok, msg = database.add_student("102", "Priya Verma", "ECE", 6, "priya@example.com")
    check("add_student 102 succeeds", ok)

    check("101 has 100.0% with 1 working day",
          database.calculate_attendance_percentage("101") == 100.0)
    check("102 has 0.0% with 1 working day",
          database.calculate_attendance_percentage("102") == 0.0)

    # Insert an old record directly, so there are 2 different working days.
    with database.db_session() as conn:
        conn.execute(
            "INSERT INTO attendance (student_id, name, date, time, status) "
            "VALUES ('102', 'Priya Verma', '2000-01-01', '09:00:00', 'Present')"
        )
    check("total working days is now 2", database.get_total_working_days() == 2)
    check("101 is now 50.0% (1 of 2 days)",
          database.calculate_attendance_percentage("101") == 50.0)
    check("102 is now 50.0% (1 of 2 days)",
          database.calculate_attendance_percentage("102") == 50.0)

    print("\n--- Searching and filtering ---")
    check("filter by student_id '102' returns 1 record",
          len(database.get_attendance(student_id="102")) == 1)
    check("filter by name 'rah' returns 1 record",
          len(database.get_attendance(name="rah")) == 1)
    check("filter by branch 'CSE' returns 1 record",
          len(database.get_attendance(branch="CSE")) == 1)
    check("get_all_students() returns 2 students",
          len(database.get_all_students()) == 2)

    print("\n--- Deleting a student ---")
    ok, msg = database.delete_student("101")
    check("delete_student('101') succeeds", ok)
    check("student 101 no longer exists", database.student_exists("101") is False)
    check("attendance rows of 101 were removed",
          database.get_attendance(student_id="101") == [])

    ok, msg = database.delete_student("999")
    check("deleting a non-existent student fails cleanly", not ok)
    print(f"        message: {msg}")


if __name__ == "__main__":
    # Start from a clean test database
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    try:
        run_tests()
    finally:
        # Always remove the temporary database, even if a test crashed
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    print("\n==============================")
    print(f"Passed: {passed}   Failed: {failed}")
    print("==============================")