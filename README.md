# Smart Attendance System Using Face Recognition

 The system uses a laptop webcam to recognize registered students and mark their attendance automatically. It stores records in SQLite, prevents duplicate attendance on the same day, and provides a Streamlit web interface with a dashboard, search, attendance percentages and CSV/Excel reports.

> **Academic demonstration only.** Face images are biometric data. This project is not production-ready biometric infrastructure. See [Privacy and security](#privacy-and-security).

---

## Features

- Administrator login (password stored as a salted PBKDF2 hash, not as plain text)
- Dashboard: Total Students, Present Today, Absent Today, Today's Attendance %, and three charts
- Student registration with validation
- Webcam face capture (an image is saved only when exactly one face is detected)
- Face detection with Haar Cascade, face recognition with LBPH
- Model training from the captured dataset
- Automatic attendance with duplicate prevention (one record per student per day)
- Attendance records with date, student ID, name and branch filters, plus sorting
- Attendance percentage per student
- CSV and Excel (.xlsx) report export
- Deleting a student also deletes their attendance records and face images
- Error handling for camera, model, dataset and database problems

## Requirements

- Windows 10 or 11
- Python 3.11
- A webcam
- Visual Studio Code (recommended)
- Packages listed in `requirements.txt` (OpenCV contrib, Streamlit, pandas, NumPy, OpenPyXL, Plotly)

## Setup (Windows, Command Prompt)

Run these in a **Command Prompt** terminal, inside the project folder.

1. Create a virtual environment:

   ```
   python -m venv venv
   ```

2. Activate it (the prompt should start with `(venv)`):

   ```
   venv\Scripts\activate
   ```

3. Install the packages:

   ```
   pip install -r requirements.txt
   ```

   Install only `opencv-contrib-python`. Do not install `opencv-python` or `opencv-python-headless` next to it, because they conflict and remove the `cv2.face` module.

4. Download the Haar Cascade file. The OpenCV 5.0.0 package used here does not include it:

   ```
   mkdir haarcascade
   curl -L -o haarcascade\haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
   ```

5. Create the database:

   ```
   python database.py
   ```

6. Check that everything is in place:

   ```
   python check_environment.py --camera
   ```

   Every line should say `[PASS]`.

## Run the application

```
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open by itself. Press `Ctrl + C` in the terminal to stop the server (closing the browser tab does not stop it).

## Login

On the first login attempt the program creates `admin_credentials.json` with the **demo** login `admin` / `admin123`. Change it before showing the project to anyone:

```
python -c "import auth, getpass; u = input('New username: '); p = getpass.getpass('New password: '); auth.create_credentials_file(u.strip(), p); print('Login updated.')"
```

To reset to the demo login, delete `admin_credentials.json`.

## Typical workflow

1. **Register Student**: enter ID, name, branch, semester and email.
2. **Capture Face**: select the student and start the camera. About 30 images are saved to `dataset\<student_id>\`.
3. **Train Model**: trains the LBPH recognizer. **Retrain after adding or deleting students.**
4. **Take Attendance**: the camera opens. Recognized students are marked Present. Press ESC to finish.
5. **Dashboard / Attendance Records / Reports**: view, filter and download attendance.

## Project structure

```
Smart_Attendance/
├── app.py                           Streamlit app (pages and navigation)
├── auth.py                          Administrator login
├── about.py                         About Project page
├── dashboard.py                     Dashboard page (metrics and charts)
├── database.py                      All SQLite operations
├── face_capture.py                  Webcam face capture
├── train_model.py                   LBPH training
├── face_recognition_attendance.py   Live recognition and attendance marking
├── utils.py                         Report helpers (tables, CSV, Excel)
├── check_environment.py             Health check before a demonstration
├── test_database.py                 Tests for database.py (uses a temporary database)
├── requirements.txt
├── README.md
├── attendance.db                    SQLite database (created automatically)
├── admin_credentials.json           Hashed login (created automatically)
├── dataset/<student_id>/            Face images (biometric data)
├── trainer/                         trainer.yml and labels.json
├── haarcascade/                     haarcascade_frontalface_default.xml
├── reports/                         Saved report copies
└── venv/                            Virtual environment
```

A `config.py` was planned originally. The settings stayed as constants at the top of the module that uses them, which keeps each file self-explanatory.

## Settings you may want to change

| Setting | File | Default | Meaning |
|---|---|---|---|
| `RECOGNITION_THRESHOLD` | `face_recognition_attendance.py` | 70.0 | LBPH **distance** limit. Lower is stricter. Raise it if real students are rejected, lower it if strangers are accepted. |
| `CONFIRM_FRAMES` | `face_recognition_attendance.py` | 5 | Consecutive frames needed before attendance is marked |
| `NUM_IMAGES` | `face_capture.py` | 30 | Images captured per student |
| `FACE_SIZE` | `train_model.py` | 200 x 200 | Faces are resized to this before training and recognition |
| `MIN_REQUIRED_PERCENT` | `utils.py` | 75.0 | Below this, the Remark column shows "Shortage" |
| `MAX_FAILED_ATTEMPTS`, `LOCKOUT_SECONDS` | `auth.py` | 5, 30 | Login lockout |

**About the recognition value:** LBPH returns a *distance*, not a probability. Lower means more similar. The "match score" shown on the camera window is just 100 minus the distance and must not be read as a probability.

**Attendance percentage** = Present Days / Total Working Days x 100, where Total Working Days is the number of different dates on which attendance was taken.

## Tests

```
python test_database.py          database functions (uses a temporary database, your data is untouched)
python utils.py                  report export test
python check_environment.py      environment health check
```

`python auth.py` runs a login self-test that expects the **demo** login (`admin` / `admin123`). It will report failures after you change the login. That is expected.

## Troubleshooting

| Message | Likely cause and fix |
|---|---|
| `Camera could not be opened...` | Another application is using the webcam, or Windows camera privacy settings block desktop apps. Close other camera apps and check Settings, Privacy and security, Camera. |
| `cv2.face` missing / `False` | A conflicting OpenCV package is installed. Uninstall all OpenCV packages, then install only `opencv-contrib-python`. |
| `Haar Cascade file not found` | Repeat setup step 4. |
| `No trained model found` | Open Train Model and train, or run `python train_model.py`. |
| `Student ID ... is not registered` | Register the student first. |
| A real student shows `Unknown Face` | Capture more images in different lighting, retrain, or raise `RECOGNITION_THRESHOLD` a little. |
| A stranger is recognized | Lower `RECOGNITION_THRESHOLD` and retrain with more students. |
| Downloaded report will not open | Windows may mark browser downloads as blocked. Right-click the file, Properties, tick Unblock, or use the "Save a copy in the reports folder" button. |

## Privacy and security

- Face images are **biometric data** and are sensitive personal information.
- Get **informed consent** from every student before capturing their face.
- Keep `dataset/`, `trainer/`, `attendance.db`, `admin_credentials.json` and any backup folders on a protected computer. Do not upload them to GitHub, shared drives or cloud sync folders.
- Restrict who can open the application and the project folder.
- Deleting a student here removes their record, attendance and face images. **Retrain the model afterwards** so it no longer contains their face data.
- A real deployment would need individual accounts, strong passwords, HTTPS, encryption of stored biometric data, audit logs, and compliance with the applicable data-protection laws.

If you use Git, add this to `.gitignore`:

```
venv/
dataset/
trainer/
reports/
backup/
attendance.db
admin_credentials.json
__pycache__/
```

## Limitations

- No liveness detection: a photo or a screen showing a face could be accepted.
- LBPH is sensitive to lighting, pose, glasses and expression, and is less accurate than modern deep-learning methods.
- "Total working days" counts the dates on which attendance was taken. There is no timetable or holiday calendar.
- One administrator account. Login is session-based, so a browser refresh signs the admin out.
- SQLite and local storage suit a single computer.
- OpenCV's on-screen text cannot draw non-English letters in names.

## Future scope

Deep-learning face embeddings, liveness detection, multiple cameras, MySQL/PostgreSQL, teacher and student roles, timetable-aware attendance, email/SMS alerts, encrypted storage, HTTPS deployment, and a mobile app.
