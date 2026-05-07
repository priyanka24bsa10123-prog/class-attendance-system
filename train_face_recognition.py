"""
train_face_recognition.py
--------------------------
Trains an LBPH (Local Binary Pattern Histogram) face recognizer
on student images stored in data/students/<id>_<name>/ folders.

Run AFTER collecting student images with add_student.py.

Usage:
    python train_face_recognition.py
"""

import cv2
import os
import numpy as np
import pickle

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/students"
MODEL_PATH  = "models/face_recognizer.yml"
LABELS_PATH = "models/face_labels.pkl"
IMG_SIZE    = (100, 100)
# ────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str):
    """
    Scans data/students/ and builds:
        faces  → list of grayscale face arrays
        labels → list of integer IDs
        id_map → {int_id: 'student_id_name'}
    """
    faces, labels = [], []
    id_map = {}
    label_counter = 0

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found. "
                                "Run add_student.py first.")

    student_folders = sorted(os.listdir(data_dir))
    if not student_folders:
        raise ValueError("No student folders found inside data/students/")

    for folder_name in student_folders:
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        id_map[label_counter] = folder_name          # e.g. "101_Alice"
        print(f"  Loading → {folder_name}  (label={label_counter})")

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(label_counter)

        label_counter += 1

    return faces, labels, id_map


def train():
    os.makedirs("models", exist_ok=True)

    print("\n[INFO] Loading student images …")
    faces, labels, id_map = load_dataset(DATA_DIR)
    print(f"[INFO] Total images : {len(faces)}")
    print(f"[INFO] Total students: {len(id_map)}")

    # LBPH is lightweight, fast, and works well with small face datasets
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

    print("\n[INFO] Training LBPH Face Recognizer …")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f"[✓] Model saved  → {MODEL_PATH}")

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(id_map, f)
    print(f"[✓] Labels saved → {LABELS_PATH}")

    print("\n[✓] Training complete!")
    print("     Students registered:")
    for k, v in id_map.items():
        print(f"       [{k}] {v}")


if __name__ == "__main__":
    print("=" * 45)
    print("    FACE RECOGNITION MODEL TRAINER")
    print("=" * 45)
    train()
