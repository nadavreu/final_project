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
        
        # Use a simple and reliable approach: get the full face bounding box
        # and extract the upper portion as the forehead region
        try:
            # Get all landmark coordinates
            all_xs = [int(landmarks[i].x * iw) for i in range(min(len(landmarks), 468))]
            all_ys = [int(landmarks[i].y * ih) for i in range(min(len(landmarks), 468))]
            
            # Calculate face bounding box
            face_x_min, face_x_max = min(all_xs), max(all_xs)
            face_y_min, face_y_max = min(all_ys), max(all_ys)
            face_width = face_x_max - face_x_min
            face_height = face_y_max - face_y_min
            
            # Calculate forehead region (upper 30% of face)
            forehead_height = int(face_height * 0.3)
            
            # Center the forehead ROI on the face center
            face_center_x = (face_x_min + face_x_max) // 2
            forehead_center_y = face_y_min + forehead_height // 2
            
            # Make the forehead ROI wider than the face for better coverage
            forehead_width = int(face_width * 0.75)  # 120% of face width
            forehead_height = int(face_height * 0.275)  # 35% of face height
            
            # Center the ROI on the face center
            x_min = max(0, face_center_x - forehead_width // 2)
            y_min = max(0, face_y_min - int(forehead_height * 0.2))  # Slightly above face
            x_max = min(iw, x_min + forehead_width)
            y_max = min(ih, y_min + forehead_height)
            
            # Recalculate final dimensions after bounds checking
            w = x_max - x_min
            h = y_max - y_min
            
            roi = [x_min, y_min, w, h]
            self.smooth_roi = self._smooth_roi(roi)
            return self.smooth_roi
            
        except (IndexError, ValueError) as e:
            # If landmark access fails, use a fallback approach
            self.logger.warning(f"Landmark access failed: {e}. Using fallback ROI.")
            return self._get_fallback_roi(frame, landmarks)
        
        try:
            # Get coordinates for all forehead landmarks
            xs = [int(landmarks[i].x * iw) for i in forehead_indices]
            ys = [int(landmarks[i].y * ih) for i in forehead_indices]
            
            # Calculate the center of the forehead landmarks
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            
            # Calculate the spread of landmarks
            x_spread = max(xs) - min(xs)
            y_spread = max(ys) - min(ys)
            
            # Define ROI size based on landmark spread with minimum sizes
            w = max(int(x_spread * 1.8), 80)  # At least 80 pixels wide
            h = max(int(y_spread * 1.5), 40)  # At least 40 pixels tall
            
            # Center the ROI on the forehead center
            x_min = max(0, center_x - w // 2)
            y_min = max(0, center_y - h // 2)
            x_max = min(iw, x_min + w)
            y_max = min(ih, y_min + h)
            
            # Recalculate final dimensions after bounds checking
            w = x_max - x_min
            h = y_max - y_min
            
            roi = [x_min, y_min, w, h]
            self.smooth_roi = self._smooth_roi(roi)
            return self.smooth_roi
            
        except (IndexError, ValueError) as e:
            # If landmark access fails, use a fallback approach
            self.logger.warning(f"Landmark access failed: {e}. Using fallback ROI.")
            return self._get_fallback_roi(frame, landmarks)
    
    def _get_fallback_roi(self, frame, landmarks):
        """Fallback method for ROI extraction when landmarks fail."""
        ih, iw, _ = frame.shape
        
        # Use a simple approach: find the top portion of the face
        # Get all landmark y-coordinates to find the top of the face
        all_ys = [int(landmarks[i].y * ih) for i in range(min(len(landmarks), 468))]
        all_xs = [int(landmarks[i].x * iw) for i in range(min(len(landmarks), 468))]
        
        # Calculate the center of the face
        face_center_x = int(np.mean(all_xs))
        face_center_y = int(np.mean(all_ys))
        
        # Calculate face dimensions
        face_width = max(all_xs) - min(all_xs)
        face_height = max(all_ys) - min(all_ys)
        
        # Define ROI size (top 30% of face width, top 25% of face height)
        w = int(face_width * 0.8)  # 80% of face width
        h = int(face_height * 0.25)  # Top 25% of face height
        
        # Center the ROI on the face center, but position it in the upper portion
        x_min = max(0, face_center_x - w // 2)
        y_min = max(0, min(all_ys) - int(h * 0.2))  # Slightly above the top of face
        x_max = min(iw, x_min + w)
        y_max = min(ih, y_min + h)
        
        # Recalculate final dimensions after bounds checking
        w = x_max - x_min
        h = y_max - y_min
        
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