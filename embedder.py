import cv2
import numpy as np

class FaceEmbedder:
    def get_embedding(self, face_img):
        face_resized = cv2.resize(face_img, (32,32))
        return face_resized.flatten().astype(float)/255.0
