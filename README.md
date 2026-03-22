#  Face Tracking & Visitor Analytics System

##  Overview

This project is a **real-time face detection, tracking, and analytics system** built using computer vision and deep learning. It detects faces from video streams, assigns unique IDs, tracks them across frames, and logs entry/exit events into a database.

The system uses **deep face embeddings** for robust identification and supports multi-video processing.

---

##  Features

*  Face detection using YOLO-based detector
*  Face tracking using CSRT tracker
*  Deep face embeddings using InsightFace
*  Unique visitor identification using cosine similarity
*  Entry & exit logging system
*  Multi-video input support
*  Automatic face image saving
*  Database integration for visitor tracking
*  Robust error handling (no crashes on empty frames)

---

##  How It Works

1. **Detection**

   * Faces are detected periodically using a YOLO-based detector.

2. **Tracking**

   * Between detections, OpenCV CSRT trackers follow faces across frames.

3. **Embedding Extraction**

   * Each detected face is passed to InsightFace to generate embeddings.

4. **Matching**

   * Cosine similarity is used to compare embeddings with known faces.

5. **Logging**

   * New faces → logged as **entry**
   * Missing faces after threshold → logged as **exit**

---

##  Project Structure

```
Face_Tracker/
│
├── main.py              # Main pipeline
├── detector.py          # Face detection (YOLO)
├── embedder.py          # Embedding generation
├── registrar.py         # Save face images
├── db_utils.py          # Database operations
├── config.json          # Configuration file
└── logs/                # Saved face images
```

---

## Configuration

Edit `config.json`:

```json
{
  "video_list": ["video1.mp4", "video2.mp4"],
  "similarity_threshold": 0.6,
  "exit_threshold_frames": 30,
  "log_dir": "logs",
  "detection_skip": 5
}
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Face_Tracker.git
cd Face_Tracker
```

### 2. Install dependencies

```bash
pip install opencv-python numpy scipy insightface
```

---

##  Usage

```bash
python main.py
```

* Press **q** to exit video window

---

##  Output

* Real-time video with bounding boxes and face IDs
* Logged events:

  ```
  [NEW] Face ID 105
  [KNOWN] Face ID 105
  [EXIT] Face ID 105
  ```
* Saved images in `logs/`
* Visitor data stored in database

---

##  Notes

* Face IDs are generated from the database (may not start from 1)
* Ensure videos exist in specified paths
* GPU support improves performance (optional)

---

##  Future Improvements

*  Real-time alert system (intruder detection)
*  Multi-camera tracking
*  Face re-identification across sessions
*  GUI dashboard
*  Faster similarity search (FAISS)

---

##  Tech Stack

* Python
* OpenCV
* InsightFace
* NumPy / SciPy
* YOLO (for detection)

---
This project is a part of a hackathon run by https://katomaran.com 
