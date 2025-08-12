import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import logging

class FaceDetector:
    def __init__(self, ema_alpha=0.6):
        self.logger = logging.getLogger('PulseVision.FaceDetector')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.ema_alpha = ema_alpha
        self.prev_roi = None
        self.smooth_roi = None
        self.smooth_history = deque(maxlen=5)

    def detect_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return landmarks
        return None

    def get_forehead_roi(self, frame, landmarks):
        ih, iw, _ = frame.shape
        forehead_indices = [10, 338, 297, 332, 284, 251]  # Example forehead region
        xs = [int(landmarks[i].x * iw) for i in forehead_indices]
        ys = [int(landmarks[i].y * ih) for i in forehead_indices]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w, h = x_max - x_min, y_max - y_min

        roi = [x_min, y_min, w, h]
        self.smooth_roi = self._smooth_roi(roi)
        return self.smooth_roi

    def _smooth_roi(self, current_roi):
        if self.prev_roi is None:
            self.prev_roi = current_roi
            return current_roi
        else:
            smoothed = [
                int(self.ema_alpha * p + (1 - self.ema_alpha) * c)
                for p, c in zip(self.prev_roi, current_roi)
            ]
            self.prev_roi = smoothed
            return smoothed

    def reset(self):
        """Reset the face detector state for a fresh measurement."""
        self.prev_roi = None
        self.smooth_roi = None
        self.smooth_history.clear()
        self.logger.debug("MediaPipe face detector reset")