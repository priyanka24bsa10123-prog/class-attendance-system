"""
add_student.py
--------------
Captures 50 face images from webcam for a new student.
Run this BEFORE training the face recognition model.

Usage:
    python add_student.py
"""

import cv2
import os

# ── Config ──────────────────────────────────────────────────────────────────
SAVE_DIR = "data/students"
SAMPLE_COUNT = 50          # Number of images to capture per student
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ────────────────────────────────────────────────────────────────────────────

def capture_student(name: str, student_id: str):
    folder = os.path.join(SAVE_DIR, f"{student_id}_{name}")
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0
    print(f"\n[INFO] Capturing {SAMPLE_COUNT} images for {name} (ID: {student_id})")
    print("[INFO] Look at the camera. Press 'q' to quit early.\n")

    while count < SAMPLE_COUNT:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            path = os.path.join(folder, f"{count}.jpg")
            cv2.imwrite(path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{SAMPLE_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capture - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[✓] Saved {count} images to '{folder}'")


if __name__ == "__main__":
    print("=" * 45)
    print("     STUDENT FACE REGISTRATION TOOL")
    print("=" * 45)
    name = input("Enter student name : ").strip()
    sid  = input("Enter student ID   : ").strip()
    if name and sid:
        capture_student(name, sid)
    else:
        print("[ERROR] Name and ID cannot be empty.")
