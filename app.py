"""
app.py
------
Streamlit interface for the Smart Attendance System.

Run with:  streamlit run app.py

Stage 3: Register Student, Students, Capture Face, Train Model,
Take Attendance, Attendance Records, Reports.
Dashboard and About Project are placeholders until the next phases.
"""

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime

import pandas as pd
import streamlit as st
import auth
import database
import face_recognition_attendance as fra
import train_model as tm
import utils  
from about import render_about_page 
from dashboard import render_dashboard                                           # NEW: report helpers
from face_capture import DATASET_DIR, count_saved_images, is_valid_student_id

# ----------------------------------------------------------------------
# Page setup (must be the FIRST Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
)

# Make sure the database tables exist (safe to run on every re-run)
database.create_database()

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PAGES = [
    "Dashboard",
    "Register Student",
    "Capture Face",
    "Train Model",
    "Take Attendance",
    "Students",
    "Attendance Records",
    "Reports",
    "About Project",
]
BRANCHES = ["CSE", "IT", "ECE", "EEE", "ME", "CE", "Other"]
SEMESTERS = list(range(1, 9))
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Small programs that are run in a separate process (they open the camera).
# sys.argv[1] and sys.argv[2] are the extra values we pass in.
CAPTURE_CODE = (
    "import sys, face_capture; "
    "ok, msg, n = face_capture.capture_faces("
    "sys.argv[1], clear_existing=(sys.argv[2] == '1')); "
    "print(msg); sys.exit(0 if ok else 1)"
)
RECOGNITION_CODE = (
    "import sys, face_recognition_attendance as r; "
    "ok, msg, marked = r.run_attendance(); "
    "print(msg); sys.exit(0 if ok else 1)"
)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def show_flash():
    """Show a message saved before a st.rerun(), then forget it."""
    flash = st.session_state.pop("flash", None)
    if flash:
        kind, text = flash            # kind is "success", "error", "warning" or "info"
        getattr(st, kind)(text)


def set_flash(kind, text):
    """Save a message to display after the next re-run."""
    st.session_state["flash"] = (kind, text)


def validate_student_form(student_id, name, semester, email):
    """Check the registration form. Returns a list of error messages (empty = OK)."""
    errors = []
    student_id = student_id.strip()
    name = name.strip()
    email = email.strip()

    if not student_id:
        errors.append("Student ID cannot be empty.")
    elif not is_valid_student_id(student_id):
        errors.append("Student ID may contain only letters, digits, '_' and '-'.")
    elif database.student_exists(student_id):
        errors.append(f"Student ID {student_id} already exists. Use a different ID.")

    if not name:
        errors.append("Student name cannot be empty.")

    if semester not in SEMESTERS:
        errors.append("Semester must be a number from 1 to 8.")

    if not email:
        errors.append("Email cannot be empty.")
    elif not EMAIL_PATTERN.match(email):
        errors.append("Email is not valid. Example: name@example.com")

    return errors


def delete_student_everything(student_id):
    """
    Delete the student from the database (including attendance records)
    AND delete the student's face-image folder from dataset/.
    Returns (success, message).
    """
    ok, message = database.delete_student(student_id)
    if not ok:
        return False, message

    folder = os.path.join(DATASET_DIR, student_id)
    if is_valid_student_id(student_id) and os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
            message += " Face images were deleted."
        except OSError as e:
            return True, (message + f" WARNING: the face-image folder {folder} "
                          f"could not be deleted: {e}")
    else:
        message += " (No face-image folder was found.)"
    return True, message


def coming_soon(page_name, phase):
    """Placeholder for pages we have not built yet."""
    st.title(page_name)
    st.info(f"This page will be built in {phase}.")


def student_options(students):
    """Return a dictionary like {'101 - Rahul Sharma': '101'} for a selectbox."""
    return {f"{s['student_id']} - {s['name']}": s["student_id"] for s in students}


def run_camera_script(code, *args):
    """
    Run a small Python program in a SEPARATE process (it opens the webcam
    window) and wait until it finishes.
    Returns (return_code, output_text). Return code 0 means success.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, *args],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError as e:
        return 1, f"Could not start the camera program: {e}"
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def model_files_exist():
    """True if both the trained model and the label file exist."""
    return os.path.exists(tm.TRAINER_PATH) and os.path.exists(tm.LABELS_PATH)


def show_sample_images(student_id, how_many=5):
    """Show a few of the student's saved face images (collapsed by default)."""
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        return
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))[:how_many]
    if files:
        with st.expander("Preview sample images"):
            st.image([os.path.join(folder, f) for f in files], width=130)


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def register_student_page():
    st.title("👤 Register Student")
    st.caption("Add a new student. Face images are captured on the next page.")
    show_flash()

    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", placeholder="e.g. 101")
            branch = st.selectbox("Branch", BRANCHES)
            email = st.text_input("Email", placeholder="e.g. rahul@example.com")
        with col2:
            name = st.text_input("Student Name", placeholder="e.g. Rahul Sharma")
            semester = st.selectbox("Semester", SEMESTERS, index=len(SEMESTERS) - 1)
        submitted = st.form_submit_button("REGISTER STUDENT")

    if submitted:
        errors = validate_student_form(student_id, name, semester, email)
        if errors:
            for error in errors:
                st.error(error)
        else:
            ok, message = database.add_student(
                student_id, name, branch, int(semester), email
            )
            if ok:
                st.success(message)
                st.info(
                    f"Next step: open **Capture Face** and capture images "
                    f"for Student ID {student_id.strip()}, then run **Train Model**."
                )
            else:
                st.error(message)


def capture_face_page():
    st.title("📷 Capture Face")
    st.caption("Capture face images of a registered student using the webcam.")

    students = database.get_all_students()
    if not students:
        st.info("No students registered yet. Register a student first.")
        return

    options = student_options(students)
    choice = st.selectbox("Select student", list(options.keys()))
    student_id = options[choice]

    existing = count_saved_images(student_id)
    st.write(f"**Face images already stored for this student:** {existing}")
    show_sample_images(student_id)

    clear_old = False
    if existing > 0:
        clear_old = st.checkbox(
            "Delete the existing images first (capture from scratch)"
        )
        if not clear_old:
            st.caption("Unticked: new images are ADDED to the existing ones.")

    st.markdown(
        "**Instructions**\n"
        "1. Click the button below. A camera window opens "
        "(check the taskbar if you cannot see it).\n"
        "2. Look at the camera, with good light on your face, "
        "and slowly turn your head a little.\n"
        "3. A green box means a face is detected and images are being saved. "
        "Only **one** face may be in view.\n"
        "4. The window closes by itself after 30 images. Press **ESC** to stop early."
    )

    if st.button("START CAMERA CAPTURE"):
        with st.spinner("Camera is running... finish or press ESC in the camera window."):
            code, output = run_camera_script(
                CAPTURE_CODE, student_id, "1" if clear_old else "0"
            )
        if code == 0:
            st.success(output)
            st.info(
                f"Images stored now: {count_saved_images(student_id)}. "
                "Next step: open **Train Model**."
            )
        else:
            st.error(output if output else "Capture failed for an unknown reason.")


def train_model_page():
    st.title("🧠 Train Model")
    st.caption("Train the LBPH face recognizer on the captured face images.")

    students = database.get_all_students()
    if not students:
        st.info("No students registered yet. Register a student first.")
        return

    # ---- Dataset summary ----
    rows = [{
        "Student ID": s["student_id"],
        "Name": s["name"],
        "Face Images": count_saved_images(s["student_id"]),
    } for s in students]
    st.subheader("Dataset summary")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    without_images = [r["Student ID"] for r in rows if r["Face Images"] == 0]
    if without_images:
        st.warning("These students have no face images and cannot be recognized: "
                   + ", ".join(without_images))

    # ---- Current model status ----
    if model_files_exist():
        trained_at = datetime.fromtimestamp(os.path.getmtime(tm.TRAINER_PATH))
        st.success(f"A trained model exists (last trained: "
                   f"{trained_at.strftime('%Y-%m-%d %H:%M:%S')}).")
    else:
        st.warning("No trained model yet. Click TRAIN MODEL below.")

    # ---- Train ----
    if st.button("TRAIN MODEL"):
        log = io.StringIO()
        with st.spinner("Training... please wait."):
            with contextlib.redirect_stdout(log):      # capture the printed log
                success, message, stats = tm.train_model()

        if success:
            st.success(message)
            col1, col2, col3 = st.columns(3)
            col1.metric("Students trained", stats["students"])
            col2.metric("Images used", stats["images"])
            col3.metric("Images skipped", stats["skipped_images"])
            if stats["students"] == 1:
                st.warning(
                    "Only ONE student was trained. LBPH will name that student for "
                    "any face, so only the distance threshold can reject strangers. "
                    "Train at least two students for a meaningful test."
                )
        else:
            st.error(message)

        with st.expander("Training log"):
            st.code(log.getvalue() or "(no log)")


def take_attendance_page():
    st.title("✅ Take Attendance")
    st.caption("Start live face recognition. Recognized students are marked Present.")

    if not model_files_exist():
        st.error("No trained model found. Open **Train Model** and train first.")
        return
    if not database.get_all_students():
        st.info("No students registered yet. Register a student first.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Recognition threshold (lower = stricter)", fra.RECOGNITION_THRESHOLD)
    col2.metric("Frames to confirm a match", fra.CONFIRM_FRAMES)

    st.markdown(
        "**Instructions**\n"
        "1. Click the button below. The camera window opens "
        "(check the taskbar if you cannot see it).\n"
        "2. Students look at the camera one at a time. "
        "The box shows *Verifying...*, then the attendance status.\n"
        "3. Unknown faces get a red box and are **never** marked.\n"
        "4. Press **ESC** in the camera window to finish."
    )

    if st.button("START ATTENDANCE CAMERA"):
        with st.spinner("Camera is running... press ESC in the camera window to finish."):
            code, output = run_camera_script(RECOGNITION_CODE)
        if code == 0:
            st.success("Attendance session finished.")
        else:
            st.error("The attendance session ended with an error.")
        if output:
            with st.expander("Session log", expanded=(code != 0)):
                st.code(output)

    # ---- Today's attendance ----
    st.divider()
    st.subheader("Today's attendance")
    today = database.get_today_attendance()
    if today:
        df = pd.DataFrame(today).rename(columns={
            "student_id": "Student ID", "name": "Name", "branch": "Branch",
            "date": "Date", "time": "Time", "status": "Status",
        })
        st.dataframe(df, hide_index=True)
    else:
        st.info("No attendance has been marked today.")


def students_page():
    st.title("🎓 Students")
    show_flash()

    students = database.get_all_students()
    if not students:
        st.info("No students registered yet. Use 'Register Student' to add one.")
        return

    # ---- Build the table ----
    df = pd.DataFrame(students).rename(columns={
        "student_id": "Student ID",
        "name": "Name",
        "branch": "Branch",
        "semester": "Semester",
        "email": "Email",
        "registration_date": "Registration Date",
    })
    df = df[["Student ID", "Name", "Branch", "Semester", "Email", "Registration Date"]]

    # ---- Search ----
    query = st.text_input("🔍 Search student",
                          placeholder="Type an ID, name, branch or email")
    if query.strip():
        mask = df.apply(
            lambda column: column.astype(str).str.contains(
                query.strip(), case=False, na=False, regex=False
            )
        ).any(axis=1)
        df = df[mask]

    st.caption(f"Showing {len(df)} of {len(students)} students")
    st.dataframe(df, hide_index=True)

    # ---- Delete ----
    st.divider()
    st.subheader("Delete a student")
    st.warning(
        "Deleting a student permanently removes:\n"
        "- the student's record,\n"
        "- all of the student's attendance records, and\n"
        "- the student's face images in the dataset folder.\n\n"
        "The **trained model still contains the old face data** until you run "
        "**Train Model** again, so retrain after deleting."
    )

    options = student_options(students)
    choice = st.selectbox("Select the student to delete", list(options.keys()))
    confirm = st.checkbox("I understand this cannot be undone")

    if st.button("DELETE STUDENT"):
        if not confirm:
            st.error("Please tick the confirmation box first.")
        else:
            ok, message = delete_student_everything(options[choice])
            if ok:
                set_flash("success", message + " Remember to retrain the model.")
            else:
                set_flash("error", message)
            st.rerun()


# ---------------------------- NEW PAGES --------------------------------
def attendance_records_page():
    st.title("📋 Attendance Records")
    st.caption("Search, filter and sort attendance records.")

    # ---- Filters ----
    col1, col2 = st.columns(2)
    with col1:
        use_date = st.checkbox("Filter by date")
        selected_date = st.date_input("Date", value=date.today(), disabled=not use_date)
        branch = st.selectbox("Branch", ["All"] + BRANCHES)
    with col2:
        student_id = st.text_input("Student ID", placeholder="Exact ID, e.g. 101")
        name = st.text_input("Student name", placeholder="Part of a name, e.g. rahul")

    records = database.get_attendance(
        date_str=selected_date.isoformat() if use_date else None,
        student_id=student_id.strip() or None,
        name=name.strip() or None,
        branch=None if branch == "All" else branch,
    )
    df = utils.build_attendance_dataframe(records, include_branch=True)

    # ---- Sorting ----
    col3, col4 = st.columns(2)
    sort_column = col3.selectbox(
        "Sort by", ["Date", "Time", "Student ID", "Name", "Branch", "Status"]
    )
    sort_order = col4.radio("Order", ["Descending", "Ascending"], horizontal=True)

    if not df.empty:
        sort_keys = ["Date", "Time"] if sort_column == "Date" else [sort_column]
        df = df.sort_values(
            by=sort_keys, ascending=(sort_order == "Ascending"), kind="stable"
        )

    # ---- Results ----
    st.divider()
    col5, col6 = st.columns(2)
    col5.metric("Records found", len(df))
    col6.metric("Different students", df["Student ID"].nunique() if not df.empty else 0)

    if df.empty:
        st.info("No attendance records match the filters.")
    else:
        st.dataframe(df, hide_index=True)
        st.caption("Tip: you can also click a column heading in the table to sort by it.")


def reports_page():
    st.title("📄 Reports")
    st.caption("Download attendance data as CSV or Excel.")

    all_records = database.get_attendance()
    if not all_records:
        st.info("No attendance records yet. Take attendance first.")
        return

    attendance_df = utils.build_attendance_dataframe(all_records)

    # ---- Report options ----
    st.subheader("Report options")
    limit_dates = st.checkbox("Limit the report to a date range")
    if limit_dates:
        all_dates = pd.to_datetime(attendance_df["Date"])
        picked = st.date_input(
            "Date range",
            value=(all_dates.min().date(), all_dates.max().date()),
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
            attendance_df = attendance_df[
                (attendance_df["Date"] >= start.isoformat())
                & (attendance_df["Date"] <= end.isoformat())
            ]
        else:
            st.info("Please select both a start date and an end date.")
            return

    include_summary = st.checkbox(
        "Include the attendance percentage as a 'Summary' sheet in the Excel file",
        value=True,
    )

    if attendance_df.empty:
        st.warning("No attendance records in the selected date range.")
        return

    # ---- Preview ----
    st.subheader("Preview")
    st.caption(f"{len(attendance_df)} attendance records")
    st.dataframe(attendance_df, hide_index=True)

    # ---- Download buttons ----
    percentage_df = utils.build_percentage_dataframe()

    csv_data = utils.dataframe_to_csv_bytes(attendance_df)
    excel_data = utils.export_to_excel_bytes(
        attendance_df, percentage_df if include_summary else None
    )
    csv_name = utils.make_report_filename("attendance_report", "csv")
    excel_name = utils.make_report_filename("attendance_report", "xlsx")

    st.subheader("Download")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇ DOWNLOAD CSV", data=csv_data,
                           file_name=csv_name, mime="text/csv")
    with col2:
        st.download_button("⬇ DOWNLOAD EXCEL", data=excel_data,
                           file_name=excel_name, mime=EXCEL_MIME)

    if st.button("SAVE A COPY IN THE REPORTS FOLDER"):
        try:
            csv_path = utils.save_report_copy(csv_data, csv_name)
            excel_path = utils.save_report_copy(excel_data, excel_name)
            st.success(f"Saved:\n- {csv_path}\n- {excel_path}")
        except OSError as e:
            st.error(f"Could not save the report files: {e}")

    # ---- Attendance percentage ----
    st.divider()
    st.subheader("Attendance percentage")
    st.caption(
        "Attendance % = Present Days / Total Working Days x 100. "
        "Total Working Days = number of different dates on which attendance was taken. "
        "This table always covers ALL dates, even if a date range is selected above. "
        f"Remark is 'Shortage' below {utils.MIN_REQUIRED_PERCENT:.0f}%."
    )
    st.dataframe(percentage_df, hide_index=True)
    st.download_button(
        "⬇ DOWNLOAD PERCENTAGE SUMMARY (CSV)",
        data=utils.dataframe_to_csv_bytes(percentage_df),
        file_name=utils.make_report_filename("attendance_percentage", "csv"),
        mime="text/csv",
    )


# ----------------------------------------------------------------------
# Main: sidebar navigation
# ----------------------------------------------------------------------
def main():
    # ---- Login gate: nothing else is shown until the admin signs in ----
    if not auth.is_logged_in():
        auth.render_login_page()
        return

    st.sidebar.title("🎓 Smart Attendance")
    st.sidebar.caption("Face Recognition Based Attendance System")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()
    auth.render_sidebar_user()
    st.sidebar.caption("B.Tech Final Year Project")

    if page == "Dashboard":
        render_dashboard()
    elif page == "Register Student":
        register_student_page()
    elif page == "Capture Face":
        capture_face_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Take Attendance":
        take_attendance_page()
    elif page == "Students":
        students_page()
    elif page == "Attendance Records":
        attendance_records_page()
    elif page == "Reports":
        reports_page()
    elif page == "About Project":
        render_about_page()

main()