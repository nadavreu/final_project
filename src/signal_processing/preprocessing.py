import numpy as np
from scipy.signal import savgol_filter, iirnotch, filtfilt

class SignalPreprocessor:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.prev_savgol_len = 15 if sampling_rate >= 30 else 9  # ~0.5 sec smoothing
        self.prev_savgol_poly = 2
        self.dynamic_notch_freqs = [1.0, 2.0, 3.0]  # Flicker noise frequencies in Hz
        self.q_factor = 30.0

    def smooth_temporal(self, signal):
        if len(signal) < self.prev_savgol_len:
            return signal  # Not enough samples
        return savgol_filter(signal, self.prev_savgol_len, self.prev_savgol_poly)

    def apply_adaptive_notch(self, signal):
        filtered = signal.copy()
        nyquist = 0.5 * self.fs
        for f0 in self.dynamic_notch_freqs:
            b, a = iirnotch(f0 / nyquist, self.q_factor)
            try:
                filtered = filtfilt(b, a, filtered)
            except Exception:
                continue  # skip if unstable
        return filtered

    def normalize_robust(self, values):
        values = np.array(values)
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        return (values - median) / mad