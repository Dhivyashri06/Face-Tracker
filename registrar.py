import os
import cv2
from datetime import datetime

class FaceRegistrar:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def save_face_image(self, face_id, face_img, event_type="entry"):
        date_str = datetime.now().strftime("%Y-%m-%d")
        folder = os.path.join(self.log_dir, date_str)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{face_id}_{event_type}_{int(datetime.now().timestamp())}.jpg")
        cv2.imwrite(path, face_img)
        return path
