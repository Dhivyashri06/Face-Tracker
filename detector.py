from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model.predict(frame, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = [int(v) for v in box]
                boxes.append((x1, y1, x2, y2))
        return boxes
