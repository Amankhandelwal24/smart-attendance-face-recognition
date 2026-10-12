"""
check_environment.py
--------------------
Health check for the Smart Attendance System.
Run it before a demonstration to find problems early.

    python check_environment.py            (normal check)
    python check_environment.py --camera   (also test the webcam briefly)

Result levels:
    PASS - fine
    WARN - not an error, but something to look at
    FAIL - something is broken and must be fixed
"""

import importlib
import importlib.metadata
import json
import os
import sqlite3
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(level, text):
    """Print one result line and count it."""
    results[level] += 1
    print(f"  [{level}] {text}")


# ----------------------------------------------------------------------
# 1. Python
# ----------------------------------------------------------------------
def check_python():
    print("\nPython")
    v = sys.version_info
    version_text = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        report("PASS", f"{version_text} (the version this project targets)")
    else:
        report("WARN", f"{version_text}. The project was built and tested on Python 3.11.")

    if sys.prefix != sys.base_prefix:
        report("PASS", "running inside a virtual environment")
    else:
        report("WARN", "NOT running inside a virtual environment. "
                       "Activate it first: venv\\Scripts\\activate")


# ----------------------------------------------------------------------
# 2. Packages
# ----------------------------------------------------------------------
def check_packages():
    """Try to import every required package. Returns the cv2 module or None."""
    print("\nPackages")
    required = [
        ("cv2", "opencv-contrib-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("plotly", "plotly"),
        ("streamlit", "streamlit"),
    ]
    cv2_module = None
    for module_name, pip_name in required:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown version")
            report("PASS", f"{pip_name} {version}")
            if module_name == "cv2":
                cv2_module = module
        except ImportError:
            report("FAIL", f"{pip_name} is not installed. Fix: pip install {pip_name}")
    return cv2_module


# ----------------------------------------------------------------------
# 3. OpenCV details
# ----------------------------------------------------------------------
def check_opencv(cv2):
    print("\nOpenCV")
    if cv2 is None:
        report("FAIL", "OpenCV could not be checked because it is not installed.")
        return

    if hasattr(cv2, "face"):
        try:
            cv2.face.LBPHFaceRecognizer_create()
            report("PASS", "cv2.face is available and the LBPH recognizer can be created")
        except cv2.error as e:
            report("FAIL", f"LBPH recognizer could not be created: {e}")
    else:
        report("FAIL", "cv2.face is missing. A conflicting OpenCV package is probably "
                       "installed. Fix: pip uninstall opencv-python opencv-python-headless "
                       "opencv-contrib-python, then pip install opencv-contrib-python")

    # Packages that clash with opencv-contrib-python
    for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless"):
        try:
            importlib.metadata.version(name)
            report("WARN", f"conflicting package installed: {name}. Uninstall it, then "
                           "reinstall opencv-contrib-python.")
        except importlib.metadata.PackageNotFoundError:
            pass


# ----------------------------------------------------------------------
# 4. Files and folders
# ----------------------------------------------------------------------
def check_files(cv2):
    print("\nProject files")
    code_files = [
        "app.py", "auth.py", "about.py", "dashboard.py", "database.py",
        "face_capture.py", "train_model.py", "face_recognition_attendance.py",
        "utils.py",
    ]
    for filename in code_files:
        if os.path.exists(os.path.join(BASE_DIR, filename)):
            report("PASS", filename)
        else:
            report("FAIL", f"{filename} is missing")

    if os.path.exists(os.path.join(BASE_DIR, "requirements.txt")):
        report("PASS", "requirements.txt")
    else:
        report("WARN", "requirements.txt is missing")

    print("\nFolders")
    for folder in ("dataset", "trainer", "haarcascade", "reports"):
        if os.path.isdir(os.path.join(BASE_DIR, folder)):
            report("PASS", f"{folder}/")
        else:
            report("WARN", f"{folder}/ does not exist yet (it is created when first needed)")

    print("\nHaar Cascade")
    cascade_path = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        report("FAIL", f"Haar Cascade file not found: {cascade_path}")
    elif cv2 is None:
        report("WARN", "cascade file exists, but it could not be loaded (OpenCV missing)")
    elif cv2.CascadeClassifier(cascade_path).empty():
        report("FAIL", "the cascade file exists but OpenCV could not load it")
    else:
        report("PASS", "haarcascade_frontalface_default.xml exists and loads")


# ----------------------------------------------------------------------
# 5. Database
# ----------------------------------------------------------------------
def check_database():
    print("\nDatabase")
    db_path = os.path.join(BASE_DIR, "attendance.db")
    if not os.path.exists(db_path):
        report("WARN", "attendance.db does not exist yet. Run: python database.py")
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            if {"students", "attendance"} <= tables:
                students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
                records = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
                report("PASS", f"attendance.db opens: {students} students, "
                               f"{records} attendance records")
            else:
                report("FAIL", f"tables missing (found: {sorted(tables)}). "
                               "Run: python database.py")
        finally:
            conn.close()
    except sqlite3.Error as e:
        report("FAIL", f"database error: {e}")


# ----------------------------------------------------------------------
# 6. Dataset and trained model
# ----------------------------------------------------------------------
def check_model():
    print("\nDataset and model")
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    if os.path.isdir(dataset_dir):
        folders = [f for f in os.listdir(dataset_dir)
                   if os.path.isdir(os.path.join(dataset_dir, f))]
        images = sum(
            len([f for f in os.listdir(os.path.join(dataset_dir, folder))
                 if f.lower().endswith(".jpg")])
            for folder in folders
        )
        if folders:
            report("PASS", f"dataset has {len(folders)} student folder(s) and {images} image(s)")
        else:
            report("WARN", "dataset is empty. Capture face images first.")

    trainer = os.path.join(BASE_DIR, "trainer", "trainer.yml")
    labels = os.path.join(BASE_DIR, "trainer", "labels.json")
    if os.path.exists(trainer) and os.path.exists(labels):
        try:
            with open(labels, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            report("PASS", f"trained model found ({count} student(s) in labels.json)")
        except (OSError, ValueError) as e:
            report("FAIL", f"labels.json cannot be read: {e}")
    elif os.path.exists(trainer) or os.path.exists(labels):
        report("FAIL", "only one of trainer.yml / labels.json exists. "
                       "Run: python train_model.py")
    else:
        report("WARN", "no trained model yet. Run: python train_model.py")


# ----------------------------------------------------------------------
# 7. Camera (optional)
# ----------------------------------------------------------------------
def check_camera(cv2):
    print("\nCamera")
    if cv2 is None:
        report("WARN", "camera not checked (OpenCV missing)")
        return
    cap = cv2.VideoCapture(0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
    try:
        if not cap.isOpened():
            report("FAIL", "Camera could not be opened. Make sure another application "
                           "is not using the webcam.")
            return
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            report("PASS", f"camera works ({width} x {height})")
        else:
            report("FAIL", "camera opened but no picture could be read")
    finally:
        cap.release()               # always free the camera


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Smart Attendance - Environment Check ===")
    check_python()
    cv2_module = check_packages()
    check_opencv(cv2_module)
    check_files(cv2_module)
    check_database()
    check_model()
    if "--camera" in sys.argv:
        check_camera(cv2_module)
    else:
        print("\nCamera: skipped (run with --camera to test the webcam)")

    print("\n==============================")
    print(f"PASS: {results['PASS']}   WARN: {results['WARN']}   FAIL: {results['FAIL']}")
    print("==============================")
    sys.exit(1 if results["FAIL"] else 0)