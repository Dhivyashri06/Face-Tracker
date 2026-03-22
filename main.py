import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import json
import os

# LOAD CONFIG 
with open("config.json") as f:
    config = json.load(f)

VIDEO_LIST = config.get("video_list", [])
SIM_THRESHOLD = config.get("similarity_threshold", 0.6)
EXIT_THRESHOLD = config.get("exit_threshold_frames", 30)
LOG_DIR = config.get("log_dir", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Default parameters for demo speed
FRAME_SKIP = 2          # skip every other frame for speed
DETECTION_SKIP = 10     # detect less frequently

app = FaceAnalysis(name='buffalo_s')  # fast model
app.prepare(ctx_id=-1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

known_embeddings = []
face_ids = []
next_id = 1

for VIDEO_PATH in VIDEO_LIST:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video {VIDEO_PATH}")
        continue

    trackers = {}             # tracker_idx -> CSRT tracker
    tracker_to_face_id = {}   # tracker_idx -> assigned face ID
    face_last_seen = {}       # face_id -> last frame seen
    last_face_image = {}      # face_id -> last cropped face image
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            continue

        # DETECTION 
        if frame_count % DETECTION_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.05, 2)

            trackers.clear()
            tracker_to_face_id.clear()
            faces = []

            for i, (x, y, w, h) in enumerate(detected_faces):
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                trackers[i] = tracker
                faces.append((i, x, y, w, h))  # unified tuple

        else:
            # TRACKING 
            faces = []
            for idx, tracker in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    faces.append((idx, x, y, w, h))

        # PROCESS FACES 
        for idx, x, y, w, h in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = x1 + w, y1 + h
            face = frame[y1:y2, x1:x2]
            if face is None or face.size == 0:
                continue

            # ASSIGN FACE ID
            if idx in tracker_to_face_id:
                matched_id = tracker_to_face_id[idx]
                print(f"[KNOWN] Face ID {matched_id}")
            else:
                # Compute embedding
                try:
                    face_resized = cv2.resize(face, (112, 112))
                    rgb_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    faces_app = app.get(rgb_face)
                    if len(faces_app) > 0:
                        embedding = faces_app[0].embedding
                        embedding = embedding / np.linalg.norm(embedding)
                    else:
                        embedding = None
                except:
                    embedding = None

                if embedding is None:
                    matched_id = next_id
                    next_id += 1
                    print(f"[NEW] Face ID {matched_id}")
                else:
                    matched_id = next_id
                    next_id += 1
                    known_embeddings.append(embedding)
                    face_ids.append(matched_id)
                    print(f"[NEW] Face ID {matched_id}")

                tracker_to_face_id[idx] = matched_id

                # SAVE ENTRY IMAGE
                save_face = cv2.resize(face, (224, 224))
                face_file = os.path.join(LOG_DIR, f"entry_{matched_id}_frame{frame_count}.jpg")
                cv2.imwrite(face_file, save_face)

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{matched_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # UPDATE LAST SEEN 
            face_last_seen[matched_id] = frame_count
            last_face_image[matched_id] = face

        # CHECK EXIT 
        for fid, last_seen in list(face_last_seen.items()):
            if frame_count - last_seen > EXIT_THRESHOLD:
                face_image = last_face_image.get(fid)
                if face_image is not None:
                    save_face = cv2.resize(face_image, (224, 224))
                    face_file = os.path.join(LOG_DIR, f"exit_{fid}_frame{frame_count}.jpg")
                    cv2.imwrite(face_file, save_face)
                print(f"[EXIT] Face ID {fid}")
                # Remove tracking
                del face_last_seen[fid]
                keys_to_remove = [k for k, v in tracker_to_face_id.items() if v == fid]
                for k in keys_to_remove:
                    del tracker_to_face_id[k]
 
        cv2.imshow(f"Face Demo - {VIDEO_PATH}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
