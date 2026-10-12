"""
dashboard.py
------------
The Dashboard page of the Smart Attendance System.

Shows today's numbers (metrics) and three Plotly charts.
app.py calls render_dashboard() when the Dashboard page is selected.
"""

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

import database
import utils

# Colours used in the charts
GREEN = "#2e9e5b"
RED = "#d64545"
GREY = "#8a8a8a"


def render_dashboard():
    st.title("📊 Dashboard")
    st.caption(
        f"Today: {date.today().strftime('%A, %d %B %Y')}.  "
        "Absent = registered student not yet marked Present today."
    )
    st.button("🔄 Refresh")        # clicking any button re-runs the page with fresh data

    students = database.get_all_students()
    if not students:
        st.info("No students registered yet. Register students to see the dashboard.")
        return

    # ------------------------------------------------------------------
    # Today's numbers
    # ------------------------------------------------------------------
    student_ids = {s["student_id"] for s in students}
    today_records = database.get_today_attendance()
    present_ids = {
        r["student_id"] for r in today_records if r["status"] == "Present"
    } & student_ids

    total_students = len(students)
    present_today = len(present_ids)
    absent_today = total_students - present_today
    percentage = round(present_today / total_students * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Present Today", present_today)
    col3.metric("Absent Today", absent_today)
    col4.metric("Today's Attendance", f"{percentage}%")

    st.divider()

    # ------------------------------------------------------------------
    # Chart 1 (left): students present per day
    # Chart 2 (right): today's attendance by branch
    # ------------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        records_df = utils.build_attendance_dataframe(database.get_attendance())
        present_df = records_df[records_df["Status"] == "Present"]
        if present_df.empty:
            st.info("No attendance has been recorded yet, so there is no daily chart.")
        else:
            daily = (
                present_df.groupby("Date").size()
                .reset_index(name="Students Present")
                .sort_values("Date")
                .tail(30)                       # at most the last 30 days
            )
            fig_daily = px.bar(
                daily, x="Date", y="Students Present", text_auto=True,
                title="Students present per day",
            )
            fig_daily.update_xaxes(type="category")
            fig_daily.update_traces(marker_color=GREEN)
            st.plotly_chart(fig_daily, key="chart_daily")

    with right:
        students_df = pd.DataFrame(students)
        students_df["branch"] = students_df["branch"].replace("", "Unknown").fillna("Unknown")
        students_df["is_present"] = students_df["student_id"].isin(present_ids)

        grouped = students_df.groupby("branch")["is_present"].agg(["sum", "count"]).reset_index()
        by_branch = pd.DataFrame({
            "Branch": grouped["branch"],
            "Present": grouped["sum"].astype(int),
            "Absent": (grouped["count"] - grouped["sum"]).astype(int),
        })
        by_branch_long = by_branch.melt(
            id_vars="Branch", var_name="Status", value_name="Students"
        )
        fig_branch = px.bar(
            by_branch_long, x="Branch", y="Students", color="Status",
            barmode="group", text_auto=True,
            color_discrete_map={"Present": GREEN, "Absent": RED},
            title="Today's attendance by branch",
        )
        st.plotly_chart(fig_branch, key="chart_branch")

    # ------------------------------------------------------------------
    # Chart 3: attendance percentage of every student
    # ------------------------------------------------------------------
    st.subheader("Attendance percentage per student")
    if database.get_total_working_days() == 0:
        st.info("No attendance has been taken yet, so percentages cannot be calculated.")
    else:
        pct_df = utils.build_percentage_dataframe()
        pct_df["Student"] = pct_df["Student ID"].astype(str) + " - " + pct_df["Name"]
        fig_pct = px.bar(
            pct_df, x="Student", y="Attendance %", color="Remark",
            color_discrete_map={"OK": GREEN, "Shortage": RED, "-": GREY},
            range_y=[0, 100], text_auto=".1f",
            title="Attendance % = Present Days / Total Working Days x 100",
        )
        fig_pct.add_hline(
            y=utils.MIN_REQUIRED_PERCENT, line_dash="dash",
            annotation_text=f"Minimum {utils.MIN_REQUIRED_PERCENT:.0f}%",
        )
        fig_pct.update_xaxes(type="category")
        st.plotly_chart(fig_pct, key="chart_percentage")

    # ------------------------------------------------------------------
    # Today's lists
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Today's lists")
    list_left, list_right = st.columns(2)

    with list_left:
        st.markdown("**Present today**")
        present_rows = [
            {"Student ID": r["student_id"], "Name": r["name"], "Time": r["time"]}
            for r in today_records
            if r["status"] == "Present" and r["student_id"] in student_ids
        ]
        if present_rows:
            st.dataframe(pd.DataFrame(present_rows), hide_index=True)
        else:
            st.info("Nobody has been marked present today.")

    with list_right:
        st.markdown("**Absent today**")
        absent_rows = [
            {"Student ID": s["student_id"], "Name": s["name"], "Branch": s["branch"]}
            for s in students
            if s["student_id"] not in present_ids
        ]
        if absent_rows:
            st.dataframe(pd.DataFrame(absent_rows), hide_index=True)
        else:
            st.success("Everyone is present today.")