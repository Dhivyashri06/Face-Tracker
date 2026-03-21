import cv2
import numpy as np
from scipy.spatial.distance import cosine
import json
from detector import FaceDetector
from registrar import FaceRegistrar
from db_utils import DBHandler
from insightface.app import FaceAnalysis

# -------------------- Load config --------------------
with open("config.json") as f:
    config = json.load(f)

VIDEO_LIST = config.get("video_list", [])
SIMILARITY_THRESHOLD = config.get("similarity_threshold", 0.6)
EXIT_THRESHOLD = config.get("exit_threshold_frames", 30)
LOG_DIR = config.get("log_dir", "logs")
DETECTION_SKIP = config.get("detection_skip", 5)

# -------------------- Initialize modules --------------------
detector = FaceDetector()
registrar = FaceRegistrar(log_dir=LOG_DIR)
db = DBHandler()

# InsightFace model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# -------------------- Face tracking data --------------------
known_embeddings = []
face_ids = []
next_face_id = 1
face_last_seen = {}    # face_id -> last frame number seen
last_face_image = {}   # face_id -> last cropped image
face_trackers = {}     # face_id -> OpenCV tracker
frame_number = 0

# -------------------- Embedding helper --------------------
def get_embedding(face_img):
    """Use InsightFace to get real face embeddings"""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    faces = app.get(face_rgb)
    if len(faces) == 0:
        return None
    embedding = faces[0].embedding
    # Normalize embedding for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# -------------------- Process each video --------------------
for VIDEO_PATH in VIDEO_LIST:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video {VIDEO_PATH}")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_frame_faces = []

        # ---------- Detection or Tracking ----------
        if frame_number % DETECTION_SKIP == 0:
            boxes = detector.detect_faces(frame)  # YOLO detection

            # Reset trackers for new detection frame
            face_trackers.clear()
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                face_trackers[idx + 1] = tracker  # temporary id, will update after embedding match

        else:
            # Update trackers for skipped frames
            boxes = []
            for face_id, tracker in list(face_trackers.items()):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    boxes.append((x, y, x + w, y + h))
                    current_frame_faces.append(face_id)

        # ---------- Process each face ----------
        for (x1, y1, x2, y2) in boxes:
            cropped_face = frame[y1:y2, x1:x2]

            embedding = get_embedding(cropped_face)
            if embedding is None:
                continue  # skip if InsightFace fails

            # Check if face is already known
            matched_id = None
            for i, emb in enumerate(known_embeddings):
                sim = 1 - cosine(embedding, emb)
                if sim > SIMILARITY_THRESHOLD:
                    matched_id = face_ids[i]
                    break

            if matched_id is None:
                # New face detected
                matched_id = next_face_id
                known_embeddings.append(embedding)
                face_ids.append(next_face_id)
                next_face_id += 1

                # Register in DB + save image
                visitor_id = db.add_visitor()
                image_path = registrar.save_face_image(visitor_id, cropped_face, event_type="entry")
                db.log_event(visitor_id, "entry", image_path)
                print(f"[NEW] Face ID {matched_id}")

            else:
                # Existing face, update last seen
                db.update_visitor_last_seen(matched_id)
                print(f"[KNOWN] Face ID {matched_id}")

            # Draw box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{matched_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Update tracking info
            face_last_seen[matched_id] = frame_number
            last_face_image[matched_id] = cropped_face
            current_frame_faces.append(matched_id)

            # Ensure tracker is updated with correct matched_id
            if matched_id not in face_trackers:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                face_trackers[matched_id] = tracker

        # ---------- Exit detection ----------
        for face_id, last_seen in list(face_last_seen.items()):
            if frame_number - last_seen > EXIT_THRESHOLD:
                image_path = registrar.save_face_image(face_id, last_face_image[face_id], event_type="exit")
                db.log_event(face_id, "exit", image_path)
                print(f"[EXIT] Face ID {face_id}")
                del face_last_seen[face_id]
                del last_face_image[face_id]
                if face_id in face_trackers:
                    del face_trackers[face_id]

        # ---------- Show video (optional) ----------
        try:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except AttributeError:
            pass  # headless environment

    cap.release()

# -------------------- Cleanup --------------------
cv2.destroyAllWindows()
total_unique = db.get_unique_visitor_count()
print(f"\nTotal Unique Visitors across all videos: {total_unique}")
db.close()
