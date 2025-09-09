import numpy as np
import time
from scipy.signal import welch, find_peaks, peak_prominences
from scipy.stats import entropy

class HeartRateEstimator:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.min_hr_hz = 40 / 60.0
        self.max_hr_hz = 180 / 60.0
        self.last_hr_freq = None
        self.freq_stability_threshold = 0.5  # Hz tolerance for frequency changes (more lenient)
        self.min_confidence_threshold = 0.4  # Lower confidence threshold to allow better convergence
        self.convergence_mode = True  # Allow more flexibility during initial convergence
        self.convergence_timeout = 30  # seconds to allow convergence
        self.start_time = time.time()

    def estimate(self, signal):
        """Estimate heart rate using frequency domain with robust peak selection."""
        try:
            # 1. Frequency domain analysis with longer window for stability
            window_size = min(512, len(signal))  # Longer window for better frequency resolution
            freqs, power = welch(signal, fs=self.fs, nperseg=window_size, nfft=2**12)
            
            # 2. Focus on heart rate band
            band_mask = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
            freqs = freqs[band_mask]
            power = power[band_mask]
            
            if len(power) == 0:
                return None, 0.0
            
            # 3. Find peaks with robust criteria
            # Use relative prominence to avoid being too sensitive
            min_prominence = 0.15 * np.max(power)  # Higher threshold for stability
            peaks, properties = find_peaks(power, distance=5, prominence=min_prominence)
            
            if len(peaks) == 0:
                return None, 0.0
            
            # 4. Score peaks based on multiple criteria
            peak_scores = []
            for i, peak_idx in enumerate(peaks):
                freq = freqs[peak_idx]
                hr_bpm = freq * 60.0
                
                # Score based on peak quality, not frequency preference
                prominence = properties['prominences'][i] / np.max(power)
                
                # Prefer peaks that are well-separated from others
                separation_score = 1.0
                for j, other_peak in enumerate(peaks):
                    if i != j:
                        other_freq = freqs[other_peak]
                        freq_diff = abs(freq - other_freq)
                        if freq_diff < 0.1:  # Penalize peaks too close to others
                            separation_score *= 0.5
                
                # Combined score - focus on peak quality, not frequency
                score = 0.7 * prominence + 0.3 * separation_score
                peak_scores.append(score)
            
            # 5. Select best peak
            best_idx = peaks[np.argmax(peak_scores)]
            hr_freq = freqs[best_idx]
            hr_bpm = hr_freq * 60.0
            
            # DIAGNOSTIC: Print all peaks for debugging
            print(f"All detected peaks:")
            for i, peak_idx in enumerate(peaks):
                freq = freqs[peak_idx]
                bpm = freq * 60.0
                prominence = properties['prominences'][i] / np.max(power)
                print(f"  Peak {i}: {bpm:.1f} BPM ({freq:.3f} Hz), prominence={prominence:.3f}, score={peak_scores[i]:.3f}")
            print(f"Selected: {hr_bpm:.1f} BPM ({hr_freq:.3f} Hz)")
            
            # 6. Calculate confidence based on peak quality
            best_score = max(peak_scores)
            confidence = min(best_score, 1.0)
            
            # 7. Enhanced stability check with physiological constraints
            current_time = time.time()
            in_convergence = current_time - self.start_time < self.convergence_timeout
            
            if self.last_hr_freq is not None:
                last_bpm = self.last_hr_freq * 60.0
                bpm_diff = abs(hr_bpm - last_bpm)
                
                # More conservative thresholds to prevent sudden jumps
                if in_convergence:
                    stability_threshold = 15  # Reduced from 25 BPM
                    confidence_threshold = 0.6  # Increased from 0.4
                    print(f"Convergence mode: allowing moderate changes ({bpm_diff:.1f} BPM)")
                else:
                    stability_threshold = 8  # Reduced from 15 BPM for better stability
                    confidence_threshold = 0.7  # Increased from 0.6
                
                # Additional physiological constraint: prevent extreme jumps
                max_physiological_change = 20  # Maximum physiologically reasonable change
                if bpm_diff > max_physiological_change:
                    print(f"Physiological constraint: {bpm_diff:.1f} BPM change > {max_physiological_change} BPM - rejecting")
                    hr_bpm = last_bpm
                    hr_freq = self.last_hr_freq
                    confidence = 0.3  # Low confidence for rejected measurement
                elif bpm_diff > stability_threshold:
                    print(f"Stability check failed: {bpm_diff:.1f} BPM change > {stability_threshold} BPM threshold")
                    if confidence < confidence_threshold:
                        print(f"Low confidence ({confidence:.2f}) - keeping previous BPM")
                        hr_bpm = last_bpm
                        hr_freq = self.last_hr_freq
                        confidence = 0.4  # Reduced confidence when keeping previous
                    else:
                        print(f"High confidence ({confidence:.2f}) - allowing BPM change")
                else:
                    print(f"BPM stable: {bpm_diff:.1f} BPM change")
            
            # Update last frequency
            self.last_hr_freq = hr_freq
            
            print(f"Frequency-domain HR: {hr_bpm:.1f} BPM, confidence: {confidence:.2f}, peaks: {len(peaks)}")
            
            return hr_bpm, confidence
            
        except Exception as e:
            print(f"Frequency-domain estimation error: {e}")
            return None, 0.0