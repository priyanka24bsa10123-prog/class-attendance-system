# 🎓 Smart Attendance System — ML-Based Face Recognition and Emotion Detection

An automated classroom attendance system using **Machine Learning** that:
- Detects and identifies students via face recognition
- Detects emotions in real-time
- Marks **Present / Absent** automatically
- Saves attendance to **CSV + Excel** with timestamps
- Operates **only between 9:30 AM – 10:00 AM** (configurable)

---

## 📁 Project Structure

```
attendance-system/
│
├── data/
│   ├── students/                  # Student face images (auto-created)
│   │   └── <ID>_<Name>/           # e.g.  101_Alice/  102_Bob/
│   │       ├── 0.jpg
│   │       ├── 1.jpg
│   │       └── ...
│   │
│   └── emotions/                  # FER-2013 dataset (download separately)
│       ├── train/
│       │   ├── angry/
│       │   ├── happy/
│       │   └── ...
│       └── test/
│           └── ...
│
├── models/                        # Saved trained models (auto-created)
│   ├── face_recognizer.yml        # LBPH face recognition model
│   ├── face_labels.pkl            # Label ↔ Student mapping
│   ├── emotion_model.h5           # CNN emotion detection model
│   └── training_history.png       # Emotion model training plot
│
├── attendance/                    # Output folder (auto-created)
│   ├── attendance_YYYY-MM-DD.csv
│   └── attendance_YYYY-MM-DD.xlsx
│
├── add_student.py                 # Register a new student (captures webcam images)
├── train_face_recognition.py      # Train LBPH face recognizer
├── train_emotion_detection.py     # Train CNN emotion model on FER-2013
├── attendance_system.py           # Main script (time-restricted)
├── test_system.py                 # Testing script (bypasses time lock)
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/attendance-system.git
cd attendance-system
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Usage

### Step 1 — Register Students
Run for **each student**. It opens the webcam and captures 50 face images.
```bash
python add_student.py
```
- Enter the student's name and ID when prompted.
- Look at the camera; images are saved to `data/students/<ID>_<Name>/`.

---

### Step 2 — Train Face Recognition Model
After registering all students:
```bash
python train_face_recognition.py
```
- Trains an **LBPH** (Local Binary Pattern Histogram) model.
- Saves model to `models/face_recognizer.yml`.

---

### Step 3 — Download & Prepare Emotion Dataset

Download **FER-2013** from Kaggle:
> https://www.kaggle.com/datasets/msambare/fer2013

Place the dataset as:
```
data/emotions/train/<emotion_class>/
data/emotions/test/<emotion_class>/
```
Emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

### Step 4 — Train Emotion Detection Model
```bash
python train_emotion_detection.py
```
- Trains a **CNN** on FER-2013 (48×48 grayscale images).
- Saves model to `models/emotion_model.h5`.
- Training may take 20–60 min depending on hardware. GPU recommended.

---

### Step 5 — Run the Attendance System

**During 9:30 AM – 10:00 AM:**
```bash
python attendance_system.py
```

**For testing (bypasses time window):**
```bash
python test_system.py --duration 30   # runs for 30 seconds
```

---

## 📊 Output Format

Attendance is saved to `attendance/attendance_YYYY-MM-DD.csv` and `.xlsx`:

| Student_ID | Student_Name | Status  | Emotion | Time_Detected       | Date       |
|------------|-------------|---------|---------|---------------------|------------|
| 1          | maya        | Present | Happy   | 2026-05-15 09:35:22 | 2026-05-15 |
| 1          | risit       | Present | Neutral | 2026-05-15 09:37:45 | 2026-05-15 |
| 1          | arya        | Absent  | N/A     | N/A                 | 2026-05-15 |

---

## 🤖 Machine Learning Models

| Component         | Technique                  | Dataset      |
|-------------------|---------------------------|--------------|
| Face Detection    | Haar Cascade (OpenCV)      | —            |
| Face Recognition  | LBPH (Own trained model)   | Custom photos |
| Emotion Detection | CNN (Own trained model)    | FER-2013     |

### Why LBPH for Face Recognition?
- Works well with **small datasets** (30–100 images per person)
- **Fast training** — no GPU required
- Robust to lighting changes
- Built into OpenCV

### CNN Architecture (Emotion)
```
Input (48×48×1)
  → Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  → Flatten → Dense(256) → BN → Dropout(0.5)
  → Dense(7) → Softmax
```

---

## 📦 Datasets Used

| Dataset | Purpose | Source | License |
|---------|---------|--------|---------|
| **FER-2013** | Emotion detection training | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) | Open (CC BY 4.0) |
| **Custom Student Photos** | Face recognition training | Captured via `add_student.py` | Your own data |

> **Note:** The Haar Cascade XML files are included with OpenCV — no separate download needed.

---

## ⚙️ Configuration

Edit the top of `attendance_system.py` to change settings:

```python
# Time window (change as needed)
START_TIME = dtime(9, 30)     # 9:30 AM
END_TIME   = dtime(10, 0)     # 10:00 AM

# Recognition sensitivity (lower = stricter matching)
CONFIDENCE_THRESHOLD = 70     # Range: 0–100
```

---

## 📋 Requirements

- Python 3.8+
- Webcam
- TensorFlow 2.x
- OpenCV with `opencv-contrib-python` (for LBPH)

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `cv2.face` not found | Install `opencv-contrib-python` instead of `opencv-python` |
| Face not recognized | Lower `CONFIDENCE_THRESHOLD` or add more student photos |
| System locked | Run `test_system.py` for testing outside time window |
| Low emotion accuracy | Train more epochs or use GPU |

---

## 📄 License
This project is for educational purposes. FER-2013 dataset license applies to the emotion model.
