"""
test_system.py
--------------
For TESTING ONLY — bypasses the 9:30–10:00 AM time window.
Runs the attendance system for a fixed duration (default 30 seconds).

Usage:
    python test_system.py [--duration 30]
"""

import cv2
import numpy as np
import pickle
import pandas as pd
import os
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FACE_MODEL_PATH    = "models/face_recognizer.yml"
LABELS_PATH        = "models/face_labels.pkl"
EMOTION_MODEL_PATH = "models/emotion_model.h5"
ATTENDANCE_DIR     = "attendance"
HAAR_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 70
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# ──────────────────────────────────────────────────────────────────────────────


def predict_emotion(model, face_gray):
    face = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
    face = np.expand_dims(face, (0, -1))
    return EMOTIONS[np.argmax(model.predict(face, verbose=0))]


def parse_student_info(folder_name):
    parts = folder_name.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (folder_name, folder_name)


def save_attendance(records, all_students):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y-%m-%d_TEST")
    detected_ids = {r["Student_ID"] for r in records}
    rows = list(records)

    for s in all_students:
        if s["id"] not in detected_ids:
            rows.append({
                "Student_ID": s["id"], "Student_Name": s["name"],
                "Status": "Absent", "Emotion": "N/A",
                "Time_Detected": "N/A", "Date": datetime.now().strftime("%Y-%m-%d")
            })

    df = pd.DataFrame(rows).sort_values("Student_ID")
    df.to_csv(f"{ATTENDANCE_DIR}/attendance_{tag}.csv", index=False)
    with pd.ExcelWriter(f"{ATTENDANCE_DIR}/attendance_{tag}.xlsx", engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Attendance")

    print(f"\n[✓] Saved → {ATTENDANCE_DIR}/attendance_{tag}.[csv|xlsx]")
    print("\n── Summary ─────────────────────────────────")
    print(df.to_string(index=False))


def run_test(duration: int):
    # Load models
    if not os.path.exists(FACE_MODEL_PATH):
        print("[ERROR] Face model missing. Run train_face_recognition.py first."); return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(FACE_MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        id_map = pickle.load(f)

    emotion_model = None
    if os.path.exists(EMOTION_MODEL_PATH):
        from tensorflow.keras.models import load_model
        emotion_model = load_model(EMOTION_MODEL_PATH)

    all_students = [
        {"id": parse_student_info(v)[0], "name": parse_student_info(v)[1]}
        for v in id_map.values()
    ]

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    cam = cv2.VideoCapture(0)
    attendance_log = {}
    start = time.time()

    print(f"[TEST] Running for {duration} seconds. Press Q to stop early.\n")

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            print("\n[INFO] Test duration ended.")
            break

        ret, frame = cam.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            fg = gray[y:y+h, x:x+w]
            label, conf = recognizer.predict(cv2.resize(fg, (100, 100)))

            if conf < CONFIDENCE_THRESHOLD and label in id_map:
                sid, sname = parse_student_info(id_map[label])
                color, display = (0, 255, 0), f"{sname} ({sid})"
            else:
                sid, sname, color, display = None, "Unknown", (0, 0, 255), "Unknown"

            emotion = predict_emotion(emotion_model, fg) if emotion_model else "N/A"

            if sid and sid not in attendance_log:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_log[sid] = {
                    "Student_ID": sid, "Student_Name": sname,
                    "Status": "Present", "Emotion": emotion,
                    "Time_Detected": ts, "Date": datetime.now().strftime("%Y-%m-%d")
                }
                print(f"  [✓] {sname} ({sid}) | Emotion: {emotion} | {ts}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        remaining = int(duration - elapsed)
        cv2.putText(frame, f"[TEST] Time left: {remaining}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Present: {len(attendance_log)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.imshow("Attendance System [TEST MODE]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    save_attendance(list(attendance_log.values()), all_students)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30,
                        help="Test duration in seconds (default: 30)")
    args = parser.parse_args()

    print("=" * 48)
    print("   ATTENDANCE SYSTEM  — TEST MODE")
    print("   (Time window restriction bypassed)")
    print("=" * 48)
    run_test(args.duration)
