"""
attendance_system.py
---------------------
Main script for the Smart Attendance System.

Features:
  ✔ Face detection  (Haar Cascade)
  ✔ Face recognition (LBPH)
  ✔ Emotion detection (CNN / FER-2013)
  ✔ Time-window lock (9:30 AM – 10:00 AM)
  ✔ Saves attendance to CSV + Excel with timestamps

Usage:
    python attendance_system.py

Notes:
  • Run add_student.py to register students first.
  • Run train_face_recognition.py to train the face model.
  • Run train_emotion_detection.py to train the emotion model.
  • Attendance is saved in the  attendance/  folder.
"""

import cv2
import numpy as np
import pickle
import pandas as pd
import os
from datetime import datetime, time as dtime
import warnings
warnings.filterwarnings("ignore")

# ── Time Window Config ───────────────────────────────────────────────────────
#    System ONLY works between START_TIME and END_TIME.
#    Change to test at a different hour if needed.
START_TIME = dtime(9, 30)    # 9:30 AM
END_TIME   = dtime(10, 0)    # 10:00 AM

# ── Paths ────────────────────────────────────────────────────────────────────
FACE_MODEL_PATH    = "models/face_recognizer.yml"
LABELS_PATH        = "models/face_labels.pkl"
EMOTION_MODEL_PATH = "models/emotion_model.h5"
ATTENDANCE_DIR     = "attendance"
HAAR_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Recognition Config ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 70    # LBPH confidence — lower = more strict (0–100)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = (48, 48)
# ────────────────────────────────────────────────────────────────────────────


def is_within_time_window() -> bool:
    """Returns True only between 9:30 AM and 10:00 AM."""
    now = datetime.now().time()
    return START_TIME <= now <= END_TIME


def load_models():
    """Load face recognizer, labels, and emotion CNN."""
    if not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(
            "Face model not found. Run train_face_recognition.py first."
        )
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "Label file not found. Run train_face_recognition.py first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(FACE_MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        id_map = pickle.load(f)

    emotion_model = None
    if os.path.exists(EMOTION_MODEL_PATH):
        from tensorflow.keras.models import load_model
        emotion_model = load_model(EMOTION_MODEL_PATH)
        print("[✓] Emotion model loaded.")
    else:
        print("[!] Emotion model not found — skipping emotion detection.")
        print("    Run train_emotion_detection.py to enable it.")

    return recognizer, id_map, emotion_model


def predict_emotion(emotion_model, face_gray: np.ndarray) -> str:
    """Predict emotion from a grayscale face crop."""
    face = cv2.resize(face_gray, IMG_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    preds = emotion_model.predict(face, verbose=0)
    return EMOTIONS[np.argmax(preds)]


def parse_student_info(folder_name: str):
    """
    Folder name format: <student_id>_<name>
    e.g. '101_Alice' → ('101', 'Alice')
    """
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return folder_name, folder_name


def get_all_students(id_map: dict) -> list:
    """Return list of all registered students as dicts."""
    students = []
    for label, folder_name in id_map.items():
        sid, name = parse_student_info(folder_name)
        students.append({"label": label, "id": sid, "name": name})
    return students


def save_attendance(records: list, all_students: list):
    """
    Merge detected records with full student list.
    Mark absent for students not detected.
    Save to CSV and Excel.
    """
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path  = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    xlsx_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.xlsx")

    detected_ids = {r["Student_ID"] for r in records}

    rows = []
    # Add detected (PRESENT) students
    for r in records:
        rows.append(r)

    # Add absent students (not detected during the session)
    for s in all_students:
        if s["id"] not in detected_ids:
            rows.append({
                "Student_ID"    : s["id"],
                "Student_Name"  : s["name"],
                "Status"        : "Absent",
                "Emotion"       : "N/A",
                "Time_Detected" : "N/A",
                "Date"          : date_str
            })

    df = pd.DataFrame(rows)
    df.sort_values("Student_ID", inplace=True)
    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")

    print(f"\n[✓] Attendance saved:")
    print(f"    CSV  → {csv_path}")
    print(f"    XLSX → {xlsx_path}")
    print("\n── Attendance Summary ──────────────────────────")
    print(df.to_string(index=False))
    print("────────────────────────────────────────────────")


def run():
    # ── Time Window Check ────────────────────────────────────────────────────
    if not is_within_time_window():
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[⛔] System is LOCKED.")
        print(f"     Current time : {now}")
        print(f"     Allowed window: {START_TIME.strftime('%I:%M %p')} "
              f"→ {END_TIME.strftime('%I:%M %p')}")
        print("\n     Please run this script during the allowed time window.")
        return

    # ── Load Models ──────────────────────────────────────────────────────────
    print("[INFO] Loading models …")
    recognizer, id_map, emotion_model = load_models()
    all_students = get_all_students(id_map)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    print(f"\n[INFO] {len(all_students)} student(s) registered.")
    print(f"[INFO] Session : {START_TIME.strftime('%I:%M %p')} → "
          f"{END_TIME.strftime('%I:%M %p')}")
    print("[INFO] Starting webcam … Press 'q' to stop.\n")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    attendance_log = {}    # {student_id: record_dict}  — one entry per student

    try:
        while True:
            # ── Re-check time on every frame ─────────────────────────────────
            if not is_within_time_window():
                print("\n[INFO] Time window ended. Closing session …")
                break

            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_gray, (100, 100))

                # ── Face Recognition ──────────────────────────────────────────
                label, confidence = recognizer.predict(face_resized)
                # LBPH: lower confidence = better match
                recognized = confidence < CONFIDENCE_THRESHOLD

                if recognized and label in id_map:
                    folder_name = id_map[label]
                    student_id, student_name = parse_student_info(folder_name)
                    display_name = f"{student_name} ({student_id})"
                    color = (0, 255, 0)   # green
                else:
                    student_id   = None
                    student_name = "Unknown"
                    display_name = "Unknown"
                    color = (0, 0, 255)   # red

                # ── Emotion Detection ─────────────────────────────────────────
                emotion = "N/A"
                if emotion_model is not None:
                    emotion = predict_emotion(emotion_model, face_gray)

                # ── Mark Attendance (once per student per session) ────────────
                if student_id and student_id not in attendance_log:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_log[student_id] = {
                        "Student_ID"    : student_id,
                        "Student_Name"  : student_name,
                        "Status"        : "Present",
                        "Emotion"       : emotion,
                        "Time_Detected" : now_str,
                        "Date"          : datetime.now().strftime("%Y-%m-%d")
                    }
                    print(f"  [✓] PRESENT → {student_name} ({student_id}) | "
                          f"Emotion: {emotion} | Time: {now_str}")

                # ── Draw on Frame ─────────────────────────────────────────────
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_name, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ── HUD overlay ──────────────────────────────────────────────────
            now_hud = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {now_hud}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Present: {len(attendance_log)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.putText(frame, "Press Q to stop", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Smart Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Manual stop by user.")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Session ended. {len(attendance_log)} student(s) marked present.")
        save_attendance(list(attendance_log.values()), all_students)


if __name__ == "__main__":
    print("=" * 48)
    print("       SMART ATTENDANCE SYSTEM  v1.0")
    print("=" * 48)
    run()
