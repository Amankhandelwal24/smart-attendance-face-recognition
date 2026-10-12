"""
face_capture.py
---------------
Captures face images of ONE registered student using the laptop webcam.

How it works:
  1. Check the student is registered in the database.
  2. Open the webcam.
  3. For every frame: convert to grayscale and detect faces (Haar Cascade).
  4. Save the image ONLY if exactly one face is found.
  5. Stop after NUM_IMAGES images, or when ESC is pressed.

Images are saved as:  dataset/<student_id>/User.<student_id>.<number>.jpg

Run:  python face_capture.py
"""

import os
import re
import time

import cv2

import database

# ----------------------------------------------------------------------
# Settings (we will move these into config.py later)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml"
)

NUM_IMAGES = 30          # how many images to capture per student
SAVE_INTERVAL = 0.3      # seconds between saved images (gives slight variety)
CAMERA_INDEX = 0         # 0 = default laptop webcam
MIN_FACE_SIZE = (100, 100)
WINDOW_NAME = "Face Capture (press ESC to stop)"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_valid_student_id(student_id):
    """Only letters, digits, underscore and hyphen are allowed (safe folder name)."""
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", student_id))


def load_face_detector():
    """
    Load the Haar Cascade face detector.
    Returns (detector, error_message). If loading fails, detector is None.
    """
    if not os.path.exists(CASCADE_PATH):
        return None, (
            "Haar Cascade file not found at:\n"
            f"  {CASCADE_PATH}\n"
            "Download haarcascade_frontalface_default.xml into the haarcascade folder."
        )
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        return None, "The Haar Cascade file exists but OpenCV could not load it."
    return detector, ""


def open_camera():
    """
    Open the webcam. On Windows the DirectShow backend (CAP_DSHOW) usually
    opens faster and releases the camera more reliably.
    Returns the camera object, or None if no camera could be opened.
    """
    backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)   # try the default backend
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def count_saved_images(student_id):
    """How many .jpg images this student already has in the dataset folder."""
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


def delete_saved_images(student_id):
    """Delete all .jpg images of this student. Returns how many were deleted."""
    folder = os.path.join(DATASET_DIR, student_id)
    deleted = 0
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpg"):
                os.remove(os.path.join(folder, filename))
                deleted += 1
    return deleted


# ----------------------------------------------------------------------
# Main capture function
# ----------------------------------------------------------------------
def capture_faces(student_id, num_images=NUM_IMAGES, clear_existing=False):
    """
    Capture face images for one registered student.
    Returns (success, message, number_of_images_saved).
    """
    student_id = str(student_id).strip()

    # ---- 1. Validate the student ID ----
    if not student_id:
        return False, "Student ID cannot be empty.", 0
    if not is_valid_student_id(student_id):
        return False, "Student ID may contain only letters, digits, '_' and '-'.", 0
    student = database.get_student(student_id)
    if student is None:
        return False, (
            f"Student ID {student_id} is not registered. "
            "Register the student first."
        ), 0

    # ---- 2. Load the face detector ----
    detector, error = load_face_detector()
    if detector is None:
        return False, error, 0

    # ---- 3. Prepare the dataset folder ----
    folder = os.path.join(DATASET_DIR, student_id)
    try:
        os.makedirs(folder, exist_ok=True)
        if clear_existing:
            delete_saved_images(student_id)
    except OSError as e:
        return False, f"Could not prepare the folder {folder}: {e}", 0

    next_number = count_saved_images(student_id) + 1

    # ---- 4. Open the camera ----
    cap = open_camera()
    if cap is None:
        return False, (
            "Camera could not be opened. Make sure another application "
            "is not using the webcam."
        ), 0

    saved = 0
    last_save_time = 0.0
    stopped_by_user = False
    camera_failed = False

    try:
        while saved < num_images:
            ret, frame = cap.read()
            if not ret or frame is None:
                camera_failed = True
                break

            # Convert to grayscale (Haar Cascade works on brightness only)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE
            )

            display = frame.copy()           # we draw on this copy, not on 'gray'
            status_text = ""
            status_color = (0, 0, 255)       # red (BGR)

            if len(faces) == 0:
                status_text = "No face detected - look at the camera"
            elif len(faces) > 1:
                status_text = "Multiple faces - only one person please"
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                (x, y, w, h) = faces[0]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status_text = "Face detected - move your head slightly"
                status_color = (0, 255, 0)   # green

                # Save only if enough time passed since the last saved image
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL:
                    filename = f"User.{student_id}.{next_number}.jpg"
                    filepath = os.path.join(folder, filename)
                    if cv2.imwrite(filepath, gray):
                        saved += 1
                        next_number += 1
                        last_save_time = now
                    else:
                        return False, f"Could not write image file: {filepath}", saved

            # Show progress text on the video window
            cv2.putText(display, f"Captured: {saved}/{num_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, status_text, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow(WINDOW_NAME, display)

            # ESC key (code 27) stops the capture
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stopped_by_user = True
                break

            # Also stop if the user closes the window with the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                stopped_by_user = True
                break

    except cv2.error as e:
        return False, f"OpenCV error while capturing: {e}", saved

    finally:
        # ALWAYS release the camera and close windows, even after an error
        cap.release()
        cv2.destroyAllWindows()

    # ---- 5. Report the result ----
    if camera_failed:
        return False, (
            "Could not read from the camera. It may have been disconnected "
            "or is being used by another application."
        ), saved
    if stopped_by_user:
        return False, f"Capture stopped early. Saved {saved} of {num_images} images.", saved
    return True, f"Capture complete. Saved {saved} images in {folder}", saved


# ----------------------------------------------------------------------
# Run this file directly:  python face_capture.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Face Capture ===")
    database.create_database()

    sid = input("Enter Student ID: ").strip()

    if not sid or not is_valid_student_id(sid):
        print("Invalid Student ID. Use only letters, digits, '_' and '-'.")
        raise SystemExit(1)

    student = database.get_student(sid)
    if student is None:
        print(f"Student ID {sid} is not registered. Register the student first.")
        raise SystemExit(1)

    print(f"Student found: {student['name']}")

    clear_old = False
    existing = count_saved_images(sid)
    if existing > 0:
        answer = input(
            f"{existing} images already exist for this student. "
            "Delete them and capture again? (y/n): "
        ).strip().lower()
        clear_old = (answer == "y")

    print("\nA camera window will open.")
    print("Look at the camera and slowly move your head a little.")
    print("Press ESC to stop early.\n")

    success, message, count = capture_faces(sid, clear_existing=clear_old)
    print(message) 