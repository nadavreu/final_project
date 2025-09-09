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
        
        # Calculate z-scores with NaN protection
        history_array = np.array(list(self.history))
        std_dev = np.std(history_array)
        
        # If standard deviation is zero (all values identical), return median
        if std_dev == 0 or np.isnan(std_dev):
            return median_bpm
        
        # Calculate z-score manually to avoid NaN issues
        mean_val = np.mean(history_array)
        z_score = (new_bpm - mean_val) / std_dev
        
        # Check for NaN or infinite values
        if np.isnan(z_score) or np.isinf(z_score):
            return median_bpm
        
        # Apply outlier rejection with stricter threshold
        if abs(z_score) > 2.0:  # Reduced from 2.5 for more aggressive filtering
            print(f"Outlier detected: BPM={new_bpm:.1f}, z-score={z_score:.2f}, using median={median_bpm:.1f}")
            return median_bpm

        return new_bpm

class ROIStabilityChecker:
    def __init__(self, min_std=3.0):  # Reduced from 5.0 to 3.0 for less sensitivity
        self.min_std = min_std

    def is_stable(self, roi_patch):
        gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        
        # Adaptive threshold based on ROI size
        # Smaller ROIs get more lenient thresholds
        roi_area = roi_patch.shape[0] * roi_patch.shape[1]
        if roi_area < 1000:  # Small ROI (like cheeks)
            adaptive_threshold = self.min_std * 0.7  # 30% more lenient
        elif roi_area < 2000:  # Medium ROI
            adaptive_threshold = self.min_std * 0.85  # 15% more lenient
        else:  # Large ROI (like forehead)
            adaptive_threshold = self.min_std
        
        return std_dev >= adaptive_threshold