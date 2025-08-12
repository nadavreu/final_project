import numpy as np
from scipy.signal import welch, find_peaks, peak_prominences
from scipy.stats import entropy

class HeartRateEstimator:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.min_hr_hz = 40 / 60.0
        self.max_hr_hz = 180 / 60.0

    def estimate(self, signal):
        freqs, power = welch(signal, fs=self.fs, nperseg=min(256, len(signal)), nfft=2**12)
        band_mask = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
        freqs = freqs[band_mask]
        power = power[band_mask]

        if len(power) == 0:
            return None, 0.0

        peaks, _ = find_peaks(power, distance=5)
        if len(peaks) == 0:
            return None, 0.0

        prominences = peak_prominences(power, peaks)[0]
        best_idx = peaks[np.argmax(prominences)]
        hr_freq = freqs[best_idx]

        # Confidence using entropy and peak shape
        psd_entropy = entropy(power / (np.sum(power) + 1e-6))
        confidence = 1.0 - psd_entropy  # Lower entropy → more confident

        return hr_freq * 60.0, max(0.0, min(1.0, confidence))