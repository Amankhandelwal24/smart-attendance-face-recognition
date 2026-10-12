"""
train_model.py
--------------
Trains the LBPH face recognizer using the images in dataset/.

Steps:
  1. Load the Haar Cascade face detector.
  2. For every student folder in dataset/ (folder name = student ID):
       - read each image in grayscale
       - detect the face and cut out the face region
       - resize it to 200 x 200 pixels
  3. Train the LBPH recognizer on all face regions.
  4. Save the model to trainer/trainer.yml
     and the label lookup table to trainer/labels.json

Run:  python train_model.py
"""

import json
import os
import time

import cv2
import numpy as np

import database
from face_capture import (
    DATASET_DIR,
    MIN_FACE_SIZE,
    is_valid_student_id,
    load_face_detector,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.json")

FACE_SIZE = (200, 200)        # every face is resized to this before training
MIN_IMAGES_WARNING = 10       # warn if a student has fewer usable images
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_face_region(gray, detector):
    """
    Detect the face in a grayscale image and return it cut out and resized.
    If several faces are found, the largest one is used.
    Returns None if no face is found.
    """
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
    face = gray[y:y + h, x:x + w]
    return cv2.resize(face, FACE_SIZE)


def train_model():
    """
    Train the model.
    Returns (success, message, stats) where stats is a dictionary with
    'students', 'images' and 'skipped_images'.
    """
    start_time = time.time()
    stats = {"students": 0, "images": 0, "skipped_images": 0}

    # ---- 1. Face detector ----
    detector, error = load_face_detector()
    if detector is None:
        return False, error, stats

    # ---- 2. Dataset folder ----
    if not os.path.isdir(DATASET_DIR):
        return False, f"Dataset folder not found: {DATASET_DIR}", stats

    student_folders = sorted(
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    )
    if not student_folders:
        return False, "The dataset folder is empty. Capture face images first.", stats

    # ---- 3. Read images and build the training data ----
    face_list = []        # the face images
    label_list = []       # a number for each face image (0, 1, 2, ...)
    label_map = {}        # number -> student ID
    next_label = 0

    print("Reading the dataset...")
    for student_id in student_folders:
        if not is_valid_student_id(student_id):
            print(f"  WARNING: skipped folder '{student_id}' (invalid name).")
            continue
        if not database.student_exists(student_id):
            print(f"  WARNING: skipped folder '{student_id}' (no such student in the database).")
            continue

        folder = os.path.join(DATASET_DIR, student_id)
        image_files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not image_files:
            print(f"  WARNING: student {student_id} has no images.")
            continue

        student_faces = []
        for filename in image_files:
            path = os.path.join(folder, filename)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:                          # missing or corrupt file
                print(f"  WARNING: could not read {path}")
                stats["skipped_images"] += 1
                continue
            face = get_face_region(gray, detector)
            if face is None:                          # no face found in this image
                stats["skipped_images"] += 1
                continue
            student_faces.append(face)

        if not student_faces:
            print(f"  WARNING: no detectable faces for student {student_id}; skipped.")
            continue

        print(f"  Student {student_id}: used {len(student_faces)} of {len(image_files)} images")
        if len(student_faces) < MIN_IMAGES_WARNING:
            print(f"  WARNING: only {len(student_faces)} usable images for {student_id}. "
                  "Capture more for better accuracy.")

        label_map[next_label] = student_id
        face_list.extend(student_faces)
        label_list.extend([next_label] * len(student_faces))
        next_label += 1

    if not face_list:
        return False, "No usable face images found. Capture face images first.", stats

    stats["students"] = len(label_map)
    stats["images"] = len(face_list)

    # ---- 4. Train and save ----
    print(f"\nTraining LBPH recognizer on {stats['images']} images "
          f"from {stats['students']} student(s)...")
    try:
        os.makedirs(TRAINER_DIR, exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(face_list, np.array(label_list, dtype=np.int32))
        recognizer.write(TRAINER_PATH)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
    except cv2.error as e:
        return False, f"OpenCV error during training: {e}", stats
    except OSError as e:
        return False, f"Could not save the model files: {e}", stats

    elapsed = time.time() - start_time
    print(f"Training took {elapsed:.1f} seconds.")

    if stats["students"] == 1:
        print("\nWARNING: only ONE student was trained. LBPH will always report that")
        print("student as the closest match, so only the distance threshold can reject")
        print("strangers. Register and capture at least one more person to test properly.")

    return True, "Training completed successfully.", stats


if __name__ == "__main__":
    print("=== Training Face Recognition Model ===")
    database.create_database()
    success, message, stats = train_model()
    print()
    print(message)
    if success:
        print(f"Students trained : {stats['students']}")
        print(f"Images used      : {stats['images']}")
        print(f"Images skipped   : {stats['skipped_images']}")
        print(f"Model file       : {TRAINER_PATH}")
        print(f"Labels file      : {LABELS_PATH}")
    else:
        raise SystemExit(1)