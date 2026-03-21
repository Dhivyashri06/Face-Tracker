import cv2
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO

VIDEO_PATH = "Data_Samples/video_sample1.mp4"
SIMILARITY_THRESHOLD = 0.6

known_embeddings = []
face_ids = []
next_face_id = 1

#YOLO model
model = YOLO('yolov8n.pt') 

def get_embedding(face_img):
    face_resized = cv2.resize(face_img, (32,32))
    return face_resized.flatten().astype(float) / 255.0

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, verbose=False)
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = x2 - x1, y2 - y1
            cropped_face = frame[y1:y2, x1:x2]
            embedding = get_embedding(cropped_face)

            # Check if face is known
            matched_id = None
            for i, emb in enumerate(known_embeddings):
                sim = 1 - cosine(embedding, emb)
                if sim > SIMILARITY_THRESHOLD:
                    matched_id = face_ids[i]
                    break

            if matched_id is None:
                matched_id = next_face_id
                known_embeddings.append(embedding)
                face_ids.append(next_face_id)
                next_face_id += 1
                print(f"[NEW] Face ID {matched_id}")
            else:
                print(f"[KNOWN] Face ID {matched_id}")

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{matched_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTotal Unique Visitors :", len(face_ids))
