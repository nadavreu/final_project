import numpy as np
import cv2
from collections import deque
from scipy.stats import zscore

class HeartRateFilter:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, new_bpm, confidence=0.5):
        if new_bpm is None:
            return None
        self.history.append(new_bpm)
        if len(self.history) < 3:
            return new_bpm

        median_bpm = np.median(self.history)
        z_scores = zscore(list(self.history))
        if abs(z_scores[-1]) > 2.5:
            return median_bpm

        return new_bpm

class ROIStabilityChecker:
    def __init__(self, min_std=5.0):
        self.min_std = min_std

    def is_stable(self, roi_patch):
        gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        return std_dev >= self.min_std