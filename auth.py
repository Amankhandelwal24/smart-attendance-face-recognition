"""
auth.py
-------
Administrator login for the Smart Attendance System.

How it works:
  * The password is NOT stored as plain text. A random salt and a
    PBKDF2-SHA256 hash are saved in admin_credentials.json.
  * The file is created automatically the first time it is needed,
    using the DEMO login below (for the academic demonstration only).
  * app.py calls is_logged_in() and render_login_page().

To reset the login to the demo values, delete admin_credentials.json.

Test the login logic (no browser needed):  python auth.py
"""

import hashlib
import hmac
import json
import os
import secrets
import time

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "admin_credentials.json")

# DEMO login for the academic project. Change it before any real use.
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

PBKDF2_ITERATIONS = 200_000      # more rounds = slower guessing for an attacker
MAX_FAILED_ATTEMPTS = 5          # wrong attempts allowed ...
LOCKOUT_SECONDS = 30             # ... before the form is locked for this long


# ----------------------------------------------------------------------
# Password hashing and credential storage
# ----------------------------------------------------------------------
def hash_password(password, salt, iterations=PBKDF2_ITERATIONS):
    """Return the PBKDF2-SHA256 hash of a password (as a hex string)."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    ).hex()


def create_credentials_file(username, password):
    """Create admin_credentials.json with a fresh random salt."""
    salt = secrets.token_bytes(16)
    data = {
        "username": username,
        "salt": salt.hex(),
        "iterations": PBKDF2_ITERATIONS,
        "password_hash": hash_password(password, salt),
    }
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_credentials():
    """Read the credentials file, creating it with the demo login if missing."""
    if not os.path.exists(CREDENTIALS_FILE):
        create_credentials_file(DEFAULT_USERNAME, DEFAULT_PASSWORD)
    with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_login(username, password):
    """
    Check a username and password.
    Returns (success, message). The message is empty on success.
    """
    try:
        creds = load_credentials()
    except (OSError, ValueError) as e:
        return False, f"Could not read the credentials file: {e}"

    try:
        stored_username = creds["username"]
        salt = bytes.fromhex(creds["salt"])
        iterations = int(creds["iterations"])
        stored_hash = creds["password_hash"]
    except (KeyError, ValueError, TypeError):
        return False, (
            "The credentials file is damaged. Delete admin_credentials.json "
            "to reset the login to the demo values."
        )

    supplied_hash = hash_password(password, salt, iterations)

    # compare_digest compares in constant time (avoids timing attacks).
    # Both checks always run, so the result does not reveal which part was wrong.
    username_ok = hmac.compare_digest(
        username.strip().encode("utf-8"), stored_username.encode("utf-8")
    )
    password_ok = hmac.compare_digest(
        supplied_hash.encode("utf-8"), stored_hash.encode("utf-8")
    )

    if username_ok and password_ok:
        return True, ""
    return False, "Invalid username or password."


# ----------------------------------------------------------------------
# Streamlit pieces (used by app.py)
# ----------------------------------------------------------------------
def is_logged_in():
    """True if the admin has signed in during this browser session."""
    return st.session_state.get("logged_in", False)


def logout():
    """Sign the admin out."""
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)


def render_login_page():
    """Show the login form. On success, mark the session as logged in."""
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.title("🎓 Smart Attendance System")
        st.subheader("Administrator Login")
        st.caption("Face Recognition Based Attendance System")

        # Brute-force protection: is the form locked right now?
        now = time.time()
        locked_until = st.session_state.get("locked_until", 0.0)
        if now < locked_until:
            remaining = int(locked_until - now) + 1
            st.error(f"Too many failed attempts. Try again in about {remaining} seconds.")
            st.button("Check again")           # clicking re-runs the page
            return

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("LOGIN")

        if submitted:
            ok, message = verify_login(username, password)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username.strip()
                st.session_state["failed_attempts"] = 0
                st.rerun()
            else:
                attempts = st.session_state.get("failed_attempts", 0) + 1
                if attempts >= MAX_FAILED_ATTEMPTS:
                    st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
                    st.session_state["failed_attempts"] = 0
                    st.error(f"Too many failed attempts. The form is locked for "
                             f"{LOCKOUT_SECONDS} seconds.")
                else:
                    st.session_state["failed_attempts"] = attempts
                    st.error(message)


def render_sidebar_user():
    """Show who is signed in, with a LOGOUT button, in the sidebar."""
    st.sidebar.write(f"👤 Signed in as **{st.session_state.get('username', 'admin')}**")
    if st.sidebar.button("LOGOUT"):
        logout()
        st.rerun()


# ----------------------------------------------------------------------
# Test: python auth.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Login logic test ===")
    passed = 0
    failed = 0

    def check(description, condition):
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {description}")
        else:
            failed += 1
            print(f"  FAIL: {description}")

    check("correct username and password are accepted",
          verify_login("admin", "admin123")[0] is True)
    check("wrong password is rejected",
          verify_login("admin", "wrong-password")[0] is False)
    check("wrong username is rejected",
          verify_login("someone", "admin123")[0] is False)
    check("empty username and password are rejected",
          verify_login("", "")[0] is False)
    check("password is case-sensitive",
          verify_login("admin", "ADMIN123")[0] is False)
    check("credentials file was created",
          os.path.exists(CREDENTIALS_FILE))

    print(f"\nPassed: {passed}   Failed: {failed}")