"""
about.py
--------
The About Project page of the Smart Attendance System.

app.py calls render_about_page() when "About Project" is selected.
Library versions and settings are read from the real installation,
so the page always matches what is actually running.
"""

import platform
import sqlite3

import cv2
import numpy as np
import openpyxl
import pandas as pd
import plotly
import streamlit as st

import utils
from face_recognition_attendance import CONFIRM_FRAMES, RECOGNITION_THRESHOLD
from train_model import FACE_SIZE


# ----------------------------------------------------------------------
# Text content (kept in lists so the page code stays short)
# ----------------------------------------------------------------------
OBJECTIVE = (
    "To build a low-cost attendance system that uses a laptop webcam to recognize "
    "registered students and mark their attendance automatically, prevents duplicate "
    "attendance on the same day, and gives the administrator records, percentages "
    "and downloadable reports."
)

FEATURES = [
    "Administrator login (password stored as a salted PBKDF2 hash)",
    "Dashboard with today's metrics and charts",
    "Student registration with input validation",
    "Student database (SQLite) with search and delete",
    "Webcam face capture, saving an image only when exactly one face is detected",
    "Face detection using Haar Cascade",
    "Face recognition using LBPH with an adjustable distance threshold",
    "Model training from the captured face dataset",
    "Automatic attendance marking in real time",
    "Duplicate attendance prevention (one record per student per day, enforced by the database)",
    "Attendance records with date, student ID, name and branch filters and sorting",
    "Attendance percentage calculation per student",
    "CSV and Excel report export",
    "Deleting a student also deletes their attendance records and face images",
    "Error handling for camera, model, dataset and database problems",
]

ADVANTAGES = [
    "Low cost: needs only a laptop and its built-in webcam",
    "Runs completely offline on an ordinary CPU, with no GPU or internet needed",
    "Training takes only a few seconds, so a new student can be added quickly",
    "Saves time compared with manual roll call and reduces human recording errors",
    "Makes proxy attendance harder than roll call (but not impossible, see Limitations)",
    "All data stays on the local machine",
    "Simple, readable code, with each module having one clear job",
    "Duplicate attendance is blocked both in the code and by a database rule",
]

LIMITATIONS = [
    "No liveness detection: a printed photo or a photo on a phone screen could be "
    "accepted as a real face",
    "LBPH is sensitive to lighting, face angle, glasses and expressions; it is less "
    "accurate than modern deep-learning methods",
    "Accuracy depends on the quality and variety of the captured images "
    "and on how the threshold is tuned",
    "The model must be retrained after adding or deleting students",
    "'Total working days' means the number of dates on which attendance was taken; "
    "there is no timetable or holiday calendar, and late registrants are measured "
    "against all days",
    "Only one administrator account, and the login is session-based, so a browser "
    "refresh signs the admin out; the demo-level lockout can be reset by refreshing",
    "SQLite and the local file storage suit one computer, not many classrooms at once",
    "OpenCV's on-screen text cannot draw non-English letters in student names",
    "Not production-ready biometric infrastructure",
]

FUTURE_SCOPE = [
    "Replace LBPH with deep-learning face embeddings (for example FaceNet or ArcFace) "
    "for higher accuracy",
    "Add liveness detection (blink or depth checks) against photo and video spoofing",
    "Support multiple cameras or classroom cameras, and group recognition",
    "Move to MySQL or PostgreSQL for several users and classrooms",
    "Add teacher and student roles, and a student portal to view their own attendance",
    "Subject-wise and period-wise attendance with a timetable and holiday calendar",
    "Email or SMS alerts for low attendance",
    "Encrypt the face dataset and model, and deploy over HTTPS",
    "Mobile app and cloud deployment",
]

PRIVACY_POINTS = [
    "Face images are **biometric data**, which is sensitive personal information.",
    "Obtain **informed consent** from every student before capturing their face.",
    "**Restrict access**: only authorised staff should be able to open the application "
    "and the project folder.",
    "**Secure storage**: keep the dataset, model and database on a protected device, "
    "and never upload them to public places such as GitHub or shared drives.",
    "Do **not expose** face images publicly. Here they are shown only to the signed-in "
    "administrator, and only on request.",
    "Provide **deletion**: deleting a student here removes their record, attendance "
    "and face images, and the model should then be retrained so it no longer "
    "contains their face data.",
    "Follow the **data-protection laws** that apply (for example India's Digital "
    "Personal Data Protection Act, 2023, or the EU's GDPR), including retention "
    "limits and the purpose of use.",
    "Use **secure authentication** in a real deployment: individual accounts, strong "
    "passwords, HTTPS and audit logs.",
    "Consider **encryption** of stored biometric data.",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def show_bullets(items):
    """Show a list of strings as a bulleted list."""
    st.markdown("\n".join(f"- {item}" for item in items))


def show_numbered(items):
    """Show a list of strings as a numbered list."""
    st.markdown("\n".join(f"{number}. {item}" for number, item in enumerate(items, start=1)))


def technology_table():
    """Technologies with the versions that are really installed."""
    rows = [
        ("Python", platform.python_version(), "Programming language"),
        ("Streamlit", st.__version__, "Web interface"),
        ("OpenCV (contrib)", cv2.__version__,
         "Webcam, Haar Cascade face detection, LBPH face recognition"),
        ("SQLite", sqlite3.sqlite_version, "Database (built into Python)"),
        ("Pandas", pd.__version__, "Tables and data processing"),
        ("NumPy", np.__version__, "Image arrays and numerical work"),
        ("OpenPyXL", openpyxl.__version__, "Excel report export"),
        ("Plotly", plotly.__version__, "Dashboard charts"),
        ("hashlib (PBKDF2)", "Python standard library", "Password hashing"),
    ]
    return pd.DataFrame(rows, columns=["Technology", "Version", "Used for"])


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------
def render_about_page():
    st.title("ℹ️ About Project")
    st.header("Smart Attendance System Using Face Recognition")
    st.caption("B.Tech Final Year Project")

    tab_overview, tab_features, tab_how, tab_eval, tab_future = st.tabs([
        "Overview", "Features", "How It Works", "Evaluation", "Future & Privacy",
    ])

    # ---------------- Overview ----------------
    with tab_overview:
        st.subheader("Project Objective")
        st.write(OBJECTIVE)
        st.subheader("Technologies Used")
        st.table(technology_table())

    # ---------------- Features ----------------
    with tab_features:
        st.subheader("Main Features")
        show_bullets(FEATURES)

    # ---------------- How it works ----------------
    with tab_how:
        st.subheader("How the System Works")
        steps = [
            "**Login:** the administrator signs in. Nothing is shown before that.",
            "**Register:** the student's ID, name, branch, semester and email are "
            "validated and stored in the SQLite `students` table. The ID must be unique.",
            "**Capture:** the webcam is opened. Each frame is converted to grayscale and a "
            "**Haar Cascade** looks for faces. An image is saved to `dataset/<student_id>/` "
            "only when **exactly one** face is found.",
            f"**Train:** the face region is cut out of each saved image and resized to "
            f"{FACE_SIZE[0]} x {FACE_SIZE[1]} pixels. The **LBPH** recognizer learns these "
            f"faces and saves `trainer.yml`, plus `labels.json` that links LBPH's numeric "
            f"labels to student IDs.",
            f"**Recognize:** in the live camera view, each detected face is resized the same "
            f"way and LBPH returns the closest student and a **distance** (lower means "
            f"more similar). The face counts as known only if the distance is at most "
            f"**{RECOGNITION_THRESHOLD:g}**.",
            f"**Confirm and mark:** the same student must be recognized in "
            f"**{CONFIRM_FRAMES} consecutive frames** before attendance is marked. The "
            f"database allows only one record per student per day, so a repeat shows "
            f"*Already Marked Present*. Unknown faces are never marked.",
            "**Report:** records can be searched and filtered, percentages are calculated, "
            "the dashboard summarises today, and reports can be downloaded as CSV or Excel.",
        ]
        show_numbered(steps)

        st.info(
            "**About the 'match score' shown on the camera window:** LBPH returns a "
            "*distance*, not a probability. The displayed score is simply 100 minus the "
            "distance (kept between 0 and 100). It is a convenience for the viewer and "
            "must not be read as 'the probability that this is the right person'."
        )
        st.markdown(
            f"**Attendance percentage** = Present Days / Total Working Days x 100. "
            f"Total Working Days = the number of different dates on which attendance was "
            f"taken. A student below **{utils.MIN_REQUIRED_PERCENT:g}%** is marked 'Shortage'."
        )

    # ---------------- Evaluation ----------------
    with tab_eval:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Advantages")
            show_bullets(ADVANTAGES)
        with col_right:
            st.subheader("Limitations")
            show_bullets(LIMITATIONS)

    # ---------------- Future scope and privacy ----------------
    with tab_future:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Future Scope")
            show_bullets(FUTURE_SCOPE)
        with col_right:
            st.subheader("Privacy Considerations")
            show_bullets(PRIVACY_POINTS)
            st.warning(
                "This is an academic demonstration. It is **not** production-ready "
                "biometric infrastructure."
            )