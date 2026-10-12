"""
utils.py
--------
Helper functions for building attendance reports (tables, CSV, Excel).

Used by the Attendance Records and Reports pages in app.py.

Run this file directly to test report creation:  python utils.py
"""

import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from openpyxl.styles import Font

import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Students below this percentage get the remark "Shortage"
MIN_REQUIRED_PERCENT = 75.0


# ----------------------------------------------------------------------
# Building tables (pandas DataFrames)
# ----------------------------------------------------------------------
def build_attendance_dataframe(records, include_branch=False):
    """
    Convert a list of attendance records (from database.get_attendance())
    into a table with clean column names.
    """
    columns = ["Student ID", "Name", "Date", "Time", "Status"]
    if include_branch:
        columns = ["Student ID", "Name", "Branch", "Date", "Time", "Status"]

    if not records:
        return pd.DataFrame(columns=columns)       # empty table with headings

    df = pd.DataFrame(records).rename(columns={
        "student_id": "Student ID",
        "name": "Name",
        "branch": "Branch",
        "date": "Date",
        "time": "Time",
        "status": "Status",
    })
    if "Branch" in df.columns:
        df["Branch"] = df["Branch"].fillna("")      # old records may have no branch
    return df[columns]


def build_percentage_dataframe():
    """
    One row per registered student:
        Present Days, Total Working Days, Attendance %, Remark

    Attendance % = Present Days / Total Working Days x 100
    Total Working Days = number of different dates with at least one record.
    """
    students = database.get_all_students()
    total_days = database.get_total_working_days()

    rows = []
    for student in students:
        student_id = student["student_id"]
        present_days = database.get_present_days(student_id)
        percentage = database.calculate_attendance_percentage(student_id)

        if total_days == 0:
            remark = "-"
        elif percentage < MIN_REQUIRED_PERCENT:
            remark = "Shortage"
        else:
            remark = "OK"

        rows.append({
            "Student ID": student_id,
            "Name": student["name"],
            "Branch": student["branch"],
            "Semester": student["semester"],
            "Present Days": present_days,
            "Total Working Days": total_days,
            "Attendance %": percentage,
            "Remark": remark,
        })

    columns = ["Student ID", "Name", "Branch", "Semester", "Present Days",
               "Total Working Days", "Attendance %", "Remark"]
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------
# Creating CSV and Excel data
# ----------------------------------------------------------------------
def dataframe_to_csv_bytes(df):
    """
    Return the table as CSV data (bytes).
    'utf-8-sig' adds a small marker so Microsoft Excel shows
    non-English letters correctly when opening the CSV.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def _format_sheet(worksheet):
    """Make an Excel sheet look neat: bold header, frozen header row, column widths."""
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    worksheet.freeze_panes = "A2"

    for column_cells in worksheet.columns:
        longest = max(
            (len(str(cell.value)) for cell in column_cells if cell.value is not None),
            default=0,
        )
        letter = column_cells[0].column_letter
        worksheet.column_dimensions[letter].width = min(max(longest + 2, 10), 40)


def export_to_excel_bytes(attendance_df, percentage_df=None):
    """
    Create an .xlsx file in memory and return it as bytes.
    Sheet 1 'Attendance' = the attendance records.
    Sheet 2 'Summary'    = attendance percentage per student (optional).
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        attendance_df.to_excel(writer, sheet_name="Attendance", index=False)
        _format_sheet(writer.sheets["Attendance"])

        if percentage_df is not None:
            percentage_df.to_excel(writer, sheet_name="Summary", index=False)
            _format_sheet(writer.sheets["Summary"])
    return buffer.getvalue()


# ----------------------------------------------------------------------
# Saving a copy in the reports/ folder
# ----------------------------------------------------------------------
def make_report_filename(prefix, extension):
    """Example: attendance_report_20261002_183015.csv"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{extension}"


def save_report_copy(data, filename):
    """
    Save report data (bytes) into the reports/ folder.
    Returns the full path of the saved file.
    Raises OSError if the file cannot be written (the caller handles it).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ----------------------------------------------------------------------
# Test: python utils.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Report export test ===")
    database.create_database()

    attendance_df = build_attendance_dataframe(database.get_attendance())
    percentage_df = build_percentage_dataframe()

    print(f"Attendance rows : {len(attendance_df)}")
    print(f"Students        : {len(percentage_df)}")
    print()
    print("Attendance percentage table:")
    print(percentage_df.to_string(index=False))
    print()

    try:
        csv_path = save_report_copy(
            dataframe_to_csv_bytes(attendance_df),
            make_report_filename("test_attendance", "csv"),
        )
        xlsx_path = save_report_copy(
            export_to_excel_bytes(attendance_df, percentage_df),
            make_report_filename("test_attendance", "xlsx"),
        )
        print(f"CSV saved   : {csv_path}")
        print(f"Excel saved : {xlsx_path}")
    except OSError as e:
        print(f"Could not save the report files: {e}")