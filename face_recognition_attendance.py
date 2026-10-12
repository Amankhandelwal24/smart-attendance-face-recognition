"""
face_recognition_attendance.py
------------------------------
Recognizes faces from the webcam and marks attendance automatically.

For every detected face:
  1. Cut out the face and resize it to 200 x 200 (same as in training).
  2. LBPH predicts the closest student and a DISTANCE (lower = better match).
  3. If distance <= RECOGNITION_THRESHOLD  -> known student.
     Otherwise                             -> "Unknown Face" (no attendance).
  4. A known student must be recognized in CONFIRM_FRAMES frames in a row
     before attendance is marked (avoids one-frame mistakes).
  5. database.mark_attendance() saves one record per student per day.

Run:  python face_recognition_attendance.py
"""

import json
import os

import cv2

import database
from face_capture import MIN_FACE_SIZE, load_face_detector, open_camera
from train_model import FACE_SIZE, LABELS_PATH, TRAINER_PATH

# ----------------------------------------------------------------------
# Settings (we will move these into config.py later)
# ----------------------------------------------------------------------
# LBPH distance limit. LOWER = stricter (fewer false accepts, more rejections).
# 70 is only a starting guess. Tune it by watching the on-screen distance.
RECOGNITION_THRESHOLD = 70.0

CONFIRM_FRAMES = 5      # consecutive matching frames needed before marking
WINDOW_NAME = "Smart Attendance (press ESC to exit)"

# Colours are in BGR order (Blue, Green, Red)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def distance_to_match_score(distance):
    """
    Convert an LBPH distance into a 0-100 'match score' for display.
    This is NOT a probability. It is just 100 minus the distance,
    limited to the range 0..100.
    """
    return int(max(0, min(100, round(100 - distance))))


def load_model():
    """
    Load trainer.yml and labels.json.
    Returns (recognizer, label_map, error_message).
    """
    if not os.path.exists(TRAINER_PATH):
        return None, None, (
            f"Trained model not found: {TRAINER_PATH}\n"
            "Train the model first:  python train_model.py"
        )
    if not os.path.exists(LABELS_PATH):
        return None, None, (
            f"Labels file not found: {LABELS_PATH}\n"
            "Train the model again:  python train_model.py"
        )

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
    except cv2.error as e:
        return None, None, f"OpenCV could not load the model: {e}"

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label_map = {int(key): value for key, value in raw.items()}
    except (OSError, ValueError) as e:
        return None, None, f"Could not read {LABELS_PATH}: {e}"

    if not label_map:
        return None, None, "labels.json is empty. Train the model again."

    return recognizer, label_map, ""


def draw_lines(frame, lines, x, y, w, h, color):
    """Write several lines of text below the face box (or above if no room)."""
    line_height = 22
    top = y + h + 22                                  # first line below the box
    if top + line_height * (len(lines) - 1) > frame.shape[0] - 30:
        top = max(22, y - 10 - line_height * (len(lines) - 1))   # put above
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x, top + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ----------------------------------------------------------------------
# Main recognition loop
# ----------------------------------------------------------------------
def run_attendance():
    """
    Run live recognition until ESC (or Q) is pressed or the window is closed.
    Returns (success, message, list_of_names_marked_in_this_session).
    """
    # ---- Load the model and the face detector ----
    recognizer, label_map, error = load_model()
    if recognizer is None:
        return False, error, []

    detector, error = load_face_detector()
    if detector is None:
        return False, error, []

    cap = open_camera()
    if cap is None:
        return False, (
            "Camera could not be opened. Make sure another application "
            "is not using the webcam."
        ), []

    streak = {}           # student_id -> consecutive frames recognized
    results = {}          # student_id -> status text shown on screen
    student_cache = {}    # student_id -> student record (avoids a DB query per frame)
    newly_marked = []     # names marked Present during this session
    camera_failed = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                camera_failed = True
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE
            )

            seen_ids = set()    # students recognized in THIS frame

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                label, distance = recognizer.predict(face)
                score = distance_to_match_score(distance)

                # ---------- Known face? ----------
                if distance <= RECOGNITION_THRESHOLD and label in label_map:
                    student_id = label_map[label]

                    if student_id not in student_cache:
                        student_cache[student_id] = database.get_student(student_id)
                    student = student_cache[student_id]

                    if student is None:
                        # In the model, but deleted from the database
                        cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)
                        draw_lines(frame, ["Not in database", "Train the model again"],
                                   x, y, w, h, RED)
                        continue

                    seen_ids.add(student_id)
                    streak[student_id] = streak.get(student_id, 0) + 1

                    # Mark attendance once the match is confirmed
                    if student_id not in results and streak[student_id] >= CONFIRM_FRAMES:
                        status, message = database.mark_attendance(student_id)
                        print(message)
                        if status == "marked":
                            results[student_id] = "PRESENT (marked now)"
                            newly_marked.append(student["name"])
                        elif status == "already_marked":
                            results[student_id] = "Already Marked Present"
                        else:
                            results[student_id] = "DATABASE ERROR"

                    status_text = results.get(
                        student_id, f"Verifying... {streak[student_id]}/{CONFIRM_FRAMES}"
                    )
                    if status_text.startswith("PRESENT"):
                        color = GREEN
                    elif status_text.startswith("Already"):
                        color = YELLOW
                    elif status_text.startswith("DATABASE"):
                        color = RED
                    else:
                        color = ORANGE

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    draw_lines(frame, [
                        f"Student ID: {student_id}",
                        f"Name: {student['name']}",
                        f"Distance: {distance:.1f}  Match score: {score}%",
                        f"Status: {status_text}",
                    ], x, y, w, h, color)

                # ---------- Unknown face ----------
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)
                    draw_lines(frame, [
                        "Unknown Face",
                        f"Distance: {distance:.1f} (limit {RECOGNITION_THRESHOLD:.0f})",
                    ], x, y, w, h, RED)

            # Forget the streak of anyone who was not recognized in this frame
            for student_id in list(streak.keys()):
                if student_id not in seen_ids:
                    del streak[student_id]

            cv2.putText(frame, "ESC = exit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):          # ESC or Q
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break                                  # window closed with X

    except cv2.error as e:
        return False, f"OpenCV error during recognition: {e}", newly_marked

    finally:
        # ALWAYS free the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

    if camera_failed:
        return False, (
            "Could not read from the camera. It may have been disconnected "
            "or is being used by another application."
        ), newly_marked

    return True, (
        f"Recognition stopped. {len(newly_marked)} new attendance "
        "record(s) marked in this session."
    ), newly_marked


# ----------------------------------------------------------------------
# Run this file directly:  python face_recognition_attendance.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Smart Attendance - Face Recognition ===")
    print(f"Recognition threshold : {RECOGNITION_THRESHOLD} (lower = stricter)")
    print(f"Frames to confirm     : {CONFIRM_FRAMES}")
    print("Look at the camera. Press ESC to exit.\n")
    database.create_database()
    success, message, marked = run_attendance()
    print(message)
    if not success:
        raise SystemExit(1)