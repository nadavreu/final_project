"""
FaceHR-style Heart Rate Estimation

This module implements the techniques used in the faceHR project for more accurate
and stable heart rate estimation with better peak detection and temporal consistency.
"""

import numpy as np
import time
from scipy.signal import welch, find_peaks, peak_prominences, butter, filtfilt
from scipy.stats import pearsonr
from collections import deque
import logging


class FaceHREstimator:
    """
    FaceHR-style heart rate estimator with improved peak detection and temporal consistency.
    
    Key improvements over standard methods:
    1. Temporal consistency tracking
    2. Multi-scale peak detection
    3. Adaptive filtering based on signal quality
    4. Robust frequency tracking
    """
    
    def __init__(self, sampling_rate=30):
        self.fs = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Heart rate frequency bounds
        self.min_hr_hz = 40 / 60.0  # 40 BPM
        self.max_hr_hz = 180 / 60.0  # 180 BPM
        
        # Temporal consistency tracking
        self.freq_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        self.last_stable_freq = None
        self.stability_count = 0
        self.min_stability_count = 3
        
        # Peak detection parameters
        self.peak_distance_min = int(self.fs * 0.4)  # Minimum 0.4 seconds between peaks
        self.peak_prominence_factor = 0.1  # Relative prominence threshold
        
        # Signal quality assessment
        self.signal_quality_history = deque(maxlen=20)
        self.min_signal_quality = 0.3
        
        # Adaptive parameters
        self.adaptive_window_size = True
        self.base_window_size = 256
        
        self.logger.info("FaceHR estimator initialized with sampling rate: %d Hz", self.fs)

    def estimate(self, signal):
        """
        Estimate heart rate using FaceHR techniques.
        
        Args:
            signal: Input PPG signal
            
        Returns:
            tuple: (heart_rate_bpm, confidence)
        """
        try:
            if len(signal) < 30:  # Need minimum samples
                return None, 0.0
            
            # 1. Assess signal quality
            signal_quality = self._assess_signal_quality(signal)
            self.signal_quality_history.append(signal_quality)
            
            if signal_quality < self.min_signal_quality:
                self.logger.debug("Low signal quality: %.3f", signal_quality)
                return self._get_fallback_estimate()
            
            # 2. Preprocess signal
            processed_signal = self._preprocess_signal(signal)
            
            # 3. Multi-scale frequency analysis
            hr_freq, confidence = self._multi_scale_frequency_analysis(processed_signal)
            
            if hr_freq is None:
                return self._get_fallback_estimate()
            
            # 4. Temporal consistency check
            hr_freq, confidence = self._apply_temporal_consistency(hr_freq, confidence)
            
            # 5. Convert to BPM
            hr_bpm = hr_freq * 60.0
            
            # 6. Update history
            self.freq_history.append(hr_freq)
            self.confidence_history.append(confidence)
            
            self.logger.debug("FaceHR estimate: %.1f BPM, confidence: %.3f, quality: %.3f", 
                            hr_bpm, confidence, signal_quality)
            
            return hr_bpm, confidence
            
        except Exception as e:
            self.logger.error("Error in FaceHR estimation: %s", str(e))
            return self._get_fallback_estimate()

    def _assess_signal_quality(self, signal):
        """Assess the quality of the input signal."""
        try:
            # 1. Signal-to-noise ratio in heart rate band
            freqs, power = welch(signal, fs=self.fs, nperseg=min(128, len(signal)//2))
            
            # Heart rate band power
            hr_mask = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
            hr_power = np.sum(power[hr_mask])
            total_power = np.sum(power)
            
            # SNR in heart rate band
            snr = hr_power / (total_power - hr_power + 1e-6)
            snr_quality = min(snr / 2.0, 1.0)
            
            # 2. Temporal consistency
            if len(signal) > 10:
                diff = np.abs(np.diff(signal))
                temporal_consistency = 1.0 / (1.0 + np.std(diff) / (np.mean(diff) + 1e-6))
            else:
                temporal_consistency = 0.5
            
            # 3. Signal amplitude stability
            signal_std = np.std(signal)
            signal_mean = np.abs(np.mean(signal))
            amplitude_stability = 1.0 / (1.0 + signal_std / (signal_mean + 1e-6))
            
            # Combined quality score
            quality = 0.4 * snr_quality + 0.3 * temporal_consistency + 0.3 * amplitude_stability
            
            return min(max(quality, 0.0), 1.0)
            
        except Exception:
            return 0.5

    def _preprocess_signal(self, signal):
        """Preprocess signal for better peak detection."""
        try:
            # 1. Remove DC component and detrend
            signal = signal - np.mean(signal)
            signal = signal - np.polyval(np.polyfit(range(len(signal)), signal, 1), range(len(signal)))
            
            # 2. Apply adaptive bandpass filter
            nyquist = self.fs / 2
            low = self.min_hr_hz / nyquist
            high = self.max_hr_hz / nyquist
            
            # Use higher order filter for better frequency separation
            b, a = butter(6, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # 3. Apply window function to reduce spectral leakage
            window = np.hanning(len(filtered))
            windowed = filtered * window
            
            return windowed
            
        except Exception:
            return signal

    def _multi_scale_frequency_analysis(self, signal):
        """Perform multi-scale frequency analysis for robust peak detection."""
        try:
            # Use multiple window sizes for different frequency resolutions
            window_sizes = [128, 256, 512] if len(signal) >= 512 else [64, 128, 256]
            window_sizes = [w for w in window_sizes if w <= len(signal)]
            
            all_peaks = []
            all_confidences = []
            
            for window_size in window_sizes:
                # Power spectral density
                freqs, power = welch(signal, fs=self.fs, nperseg=window_size, nfft=window_size*2)
                
                # Focus on heart rate band
                band_mask = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
                hr_freqs = freqs[band_mask]
                hr_power = power[band_mask]
                
                if len(hr_power) == 0:
                    continue
                
                # Find peaks with adaptive parameters
                min_prominence = self.peak_prominence_factor * np.max(hr_power)
                peaks, properties = find_peaks(hr_power, 
                                             distance=max(1, int(len(hr_power) * 0.1)),
                                             prominence=min_prominence)
                
                if len(peaks) > 0:
                    # Score peaks based on prominence and frequency consistency
                    for i, peak_idx in enumerate(peaks):
                        freq = hr_freqs[peak_idx]
                        prominence = properties['prominences'][i]
                        
                        # Calculate confidence based on peak quality
                        peak_confidence = prominence / np.max(hr_power)
                        
                        # Bonus for frequency consistency with previous estimates
                        if self.last_stable_freq is not None:
                            freq_diff = abs(freq - self.last_stable_freq)
                            consistency_bonus = max(0, 1.0 - freq_diff / 0.5)  # 0.5 Hz tolerance
                            peak_confidence *= (1.0 + 0.2 * consistency_bonus)
                        
                        all_peaks.append(freq)
                        all_confidences.append(peak_confidence)
            
            if not all_peaks:
                return None, 0.0
            
            # Select the most confident peak
            best_idx = np.argmax(all_confidences)
            best_freq = all_peaks[best_idx]
            best_confidence = all_confidences[best_idx]
            
            return best_freq, min(best_confidence, 1.0)
            
        except Exception as e:
            self.logger.error("Error in multi-scale analysis: %s", str(e))
            return None, 0.0

    def _apply_temporal_consistency(self, hr_freq, confidence):
        """Apply temporal consistency to prevent frequency jumping."""
        try:
            if self.last_stable_freq is None:
                self.last_stable_freq = hr_freq
                self.stability_count = 1
                return hr_freq, confidence
            
            # Check if frequency is consistent with recent history
            freq_diff = abs(hr_freq - self.last_stable_freq)
            max_freq_change = 0.3  # 0.3 Hz (18 BPM) maximum change
            
            if freq_diff <= max_freq_change:
                # Frequency is consistent
                self.stability_count += 1
                
                # Update stable frequency with weighted average
                alpha = 0.3  # Learning rate
                self.last_stable_freq = (1 - alpha) * self.last_stable_freq + alpha * hr_freq
                
                # Boost confidence for consistent estimates
                if self.stability_count >= self.min_stability_count:
                    confidence = min(confidence * 1.2, 1.0)
                
                return self.last_stable_freq, confidence
            else:
                # Frequency jump detected
                if confidence > 0.8:  # High confidence - allow the change
                    self.logger.debug("High confidence frequency change: %.3f -> %.3f Hz", 
                                    self.last_stable_freq, hr_freq)
                    self.last_stable_freq = hr_freq
                    self.stability_count = 1
                    return hr_freq, confidence
                else:
                    # Low confidence - keep previous frequency
                    self.logger.debug("Rejecting low confidence frequency change: %.3f -> %.3f Hz", 
                                    self.last_stable_freq, hr_freq)
                    self.stability_count = max(0, self.stability_count - 1)
                    return self.last_stable_freq, confidence * 0.8
            
        except Exception as e:
            self.logger.error("Error in temporal consistency: %s", str(e))
            return hr_freq, confidence

    def _get_fallback_estimate(self):
        """Get fallback estimate when primary method fails."""
        if self.last_stable_freq is not None:
            # Use last stable frequency with reduced confidence
            hr_bpm = self.last_stable_freq * 60.0
            confidence = 0.3  # Low confidence for fallback
            return hr_bpm, confidence
        else:
            return None, 0.0

    def reset(self):
        """Reset the estimator state."""
        self.freq_history.clear()
        self.confidence_history.clear()
        self.signal_quality_history.clear()
        self.last_stable_freq = None
        self.stability_count = 0
        self.logger.debug("FaceHR estimator reset")

    def get_statistics(self):
        """Get estimation statistics."""
        if not self.freq_history:
            return {}
        
        return {
            'avg_confidence': np.mean(self.confidence_history),
            'stability_count': self.stability_count,
            'signal_quality': np.mean(self.signal_quality_history) if self.signal_quality_history else 0.0,
            'freq_consistency': 1.0 - np.std(self.freq_history) / (np.mean(self.freq_history) + 1e-6)
        }
