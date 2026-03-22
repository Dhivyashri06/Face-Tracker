import cv2
import numpy as np

class FaceEmbedder:
    def get_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            print("Invalid face image")
            return None

        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (32, 32))
            embedding = face_resized.flatten().astype(float) / 255.0
            return embedding

        except Exception as e:
            print("Error in embedding:", e)
            return None
