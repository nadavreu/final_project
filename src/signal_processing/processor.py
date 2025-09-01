# import numpy as np
# from scipy import signal
# from sklearn.decomposition import FastICA
# import cv2
# from collections import deque
# import time
# import logging

# class SignalProcessor:
#     def __init__(self, sampling_rate=30):
#         """Initialize the signal processor."""
#         self.sampling_rate = sampling_rate
#         self.window_size = int(sampling_rate * 10)  # 10 seconds window
        
#         # Reduced thresholds for more lenient signal quality assessment
#         self.min_bpm_confidence = 0.5
#         self.signal_quality_threshold = 0.005
        
#         # Heart rate parameters
#         self.min_hr = 40
#         self.max_hr = 180
        
#         # Initialize filters
#         self.setup_filters()
        
#         # Signal processing parameters
#         self.window = signal.windows.hann(self.window_size)
#         self.smooth_hr = None
#         self.no_update_count = 0
#         self.max_no_update = 15  # Increased from 10
        
#         # BPM update control
#         self.last_update_time = time.time()
#         self.update_interval = 0.1
#         self.bpm_history = deque(maxlen=5)
        
#         self.baseline = None
#         self.values_history = []
#         self.debug = True
        
#         # Configure logging
#         self.logger = logging.getLogger(__name__)
        
#         # Buffer parameters
#         self.min_samples = 30  # Reduced from 60 to 30 (1 second of data)
        
#         # Initialize signal buffer
#         self.green_values = deque(maxlen=self.window_size)
        
#         # Cache for intermediate results
#         self.last_bpm = None
#         self.last_confidence = None
#         self.fft_freqs = np.fft.rfftfreq(self.window_size, d=1.0/self.sampling_rate)
#         self.valid_freq_mask = (self.fft_freqs >= self.min_hr/60) & (self.fft_freqs <= self.max_hr/60)
        
#         self.logger.info("SignalProcessor initialized with sampling rate: %d Hz", self.sampling_rate)

#     def setup_filters(self):
#         """Initialize all signal processing filters."""
#         nyquist = self.sampling_rate / 2
        
#         # Single bandpass filter for heart rate frequency range (0.75-3 Hz)
#         low = self.min_hr / 60 / nyquist
#         high = self.max_hr / 60 / nyquist
#         self.b_bandpass, self.a_bandpass = signal.butter(3, [low, high], btype='band')
        
#         # Single notch filter at 1 Hz (common noise frequency)
#         self.b_notch, self.a_notch = signal.iirnotch(1.0 / nyquist, 30.0)
        
#         # Moving average filter length
#         self.ma_length = 5

#     def preprocess_frame(self, frame, roi):
#         """Extract and preprocess the green channel from ROI."""
#         if frame is None or roi is None:
#             self.logger.warning("Invalid frame or ROI received")
#             return None
            
#         try:
#             x, y, w, h = roi
#             if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
#                 self.logger.warning("ROI coordinates out of frame bounds: (%d,%d,%d,%d)", x, y, w, h)
#                 return None
                
#             roi_frame = frame[y:y+h, x:x+w]
#             if roi_frame.size == 0:
#                 self.logger.warning("Empty ROI frame detected")
#                 return None
            
#             # Extract green channel
#             green = roi_frame[:, :, 1].astype(float)
            
#             # Apply CLAHE for contrast enhancement
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             green = clahe.apply(green.astype(np.uint8))
            
#             # Apply spatial averaging with weights
#             weights = np.ones_like(green) / green.size
#             green_mean = np.average(green, weights=weights)
            
#             self.logger.debug("Frame preprocessed - Green channel mean: %.4f", green_mean)
#             return green_mean
            
#         except Exception as e:
#             self.logger.error("Error in preprocess_frame: %s", str(e))
#             return None

#     def update_signal(self, value):
#         """Update the signal buffer with a new value."""
#         try:
#             self.green_values.append(value)
#             self.logger.debug("Signal buffer updated - Buffer size: %d", len(self.green_values))
#         except Exception as e:
#             self.logger.error("Error updating signal buffer: %s", str(e))

#     def apply_filters(self, data):
#         """Apply all filters to the signal."""
#         # Detrend and normalize
#         data = signal.detrend(data)
#         data = (data - np.mean(data)) / np.std(data)
        
#         # Apply all notch filters
#         for b, a in self.notch_filters:
#             data = signal.filtfilt(b, a, data)
        
#         # Apply bandpass filter
#         data = signal.filtfilt(self.b_bandpass, self.a_bandpass, data)
        
#         # Create window of correct size
#         window = signal.windows.hann(len(data))
        
#         # Apply window function
#         data = data * window
        
#         # Additional smoothing
#         data = signal.filtfilt(self.b_lowpass, self.a_lowpass, data)
        
#         return data

#     def extract_components(self, data):
#         """Apply ICA to extract independent components."""
#         try:
#             # Create multiple phase-shifted signals
#             shifts = [0, 1, 2, 3, 4]
#             X = np.column_stack([np.roll(data, shift) for shift in shifts])
            
#             # Apply ICA
#             ica = FastICA(n_components=len(shifts), random_state=42, max_iter=1000)
#             components = ica.fit_transform(X)
            
#             # Select the component with highest periodicity
#             periodicities = [self.estimate_periodicity(comp) for comp in components.T]
#             best_idx = np.argmax(periodicities)
            
#             # Check if the best component is good enough
#             if periodicities[best_idx] > self.signal_quality_threshold:
#                 return components[:, best_idx]
#             return data
#         except:
#             return data

#     def estimate_periodicity(self, data):
#         """Estimate the periodicity of a signal component."""
#         # Calculate autocorrelation
#         corr = np.correlate(data, data, mode='full')
#         corr = corr[len(corr)//2:]
        
#         # Find peaks in autocorrelation
#         peaks, properties = signal.find_peaks(corr, distance=10)
#         if len(peaks) < 2:
#             return 0
        
#         # Calculate peak quality metrics
#         peak_heights = corr[peaks]
#         peak_distances = np.diff(peaks)
        
#         # Assess periodicity quality
#         height_consistency = 1.0 - np.std(peak_heights) / np.mean(peak_heights)
#         distance_consistency = 1.0 - np.std(peak_distances) / np.mean(peak_distances)
        
#         return height_consistency * distance_consistency

#     def find_heart_rate(self, data):
#         """Calculate heart rate from frequency analysis."""
#         # Calculate FFT
#         fft_data = np.fft.rfft(data)
#         freqs = np.fft.rfftfreq(len(data), d=1/self.sampling_rate)
        
#         # Find peaks in the frequency domain
#         magnitude = np.abs(fft_data)
#         freq_mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#         valid_freqs = freqs[freq_mask]
#         valid_magnitude = magnitude[freq_mask]
        
#         if len(valid_magnitude) == 0:
#             return None, 0
        
#         # Normalize magnitude
#         valid_magnitude = valid_magnitude / np.max(valid_magnitude)
        
#         # Find peaks in the magnitude spectrum with adjusted parameters
#         peaks, properties = signal.find_peaks(valid_magnitude, 
#                                             distance=8,  # Reduced for more peaks
#                                             height=0.15,
#                                             prominence=0.1)
#         if len(peaks) == 0:
#             return None, 0
        
#         # Calculate confidence based on multiple factors
#         peak_heights = valid_magnitude[peaks]
#         background = np.mean(valid_magnitude)
        
#         # Calculate signal quality metrics
#         snr = np.max(peak_heights) / background if background > 0 else 0
#         peak_ratio = np.max(peak_heights) / np.sum(peak_heights)
#         periodicity = self.estimate_periodicity(data)
        
#         # Dynamic confidence calculation
#         if self.no_update_count > 5:
#             # Give more weight to SNR when we haven't updated in a while
#             confidence = (0.4 * snr + 0.3 * peak_ratio + 0.3 * periodicity)
#         else:
#             # Normal weights
#             confidence = (0.3 * snr + 0.3 * peak_ratio + 0.4 * periodicity)
        
#         # Find the most prominent peak
#         max_idx = peaks[np.argmax(peak_heights)]
        
#         # Use quadratic interpolation for better frequency estimation
#         if 0 < max_idx < len(valid_magnitude) - 1:
#             try:
#                 true_freq = self.quadratic_peak_interp(
#                     valid_freqs[max_idx-1:max_idx+2],
#                     valid_magnitude[max_idx-1:max_idx+2]
#                 )
#             except:
#                 true_freq = valid_freqs[max_idx]
#         else:
#             true_freq = valid_freqs[max_idx]
        
#         heart_rate = true_freq * 60
        
#         if self.debug:
#             print(f"SNR: {snr:.2f}, Peak Ratio: {peak_ratio:.2f}, Periodicity: {periodicity:.2f}")
#             print(f"Frequency Resolution: {freqs[1]-freqs[0]:.3f} Hz")
#             if self.no_update_count > 0:
#                 print(f"No update count: {self.no_update_count}")
        
#         return heart_rate, confidence

#     def quadratic_peak_interp(self, freqs, magnitudes):
#         """Quadratic interpolation for peak frequency estimation."""
#         alpha = magnitudes[0]
#         beta = magnitudes[1]
#         gamma = magnitudes[2]
#         peak_pos = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
#         return freqs[1] + peak_pos * (freqs[1] - freqs[0])

#     def is_valid_bpm(self, bpm):
#         """Check if BPM value is physiologically plausible."""
#         if bpm is None:
#             return False
            
#         # Check if within physiological range
#         if not (self.min_hr <= bpm <= self.max_hr):
#             return False
            
#         # Check for sudden changes
#         if self.bpm_history:
#             last_bpm = self.bpm_history[-1]
#             max_change = min(25, 15 + self.no_update_count)
#             if abs(bpm - last_bpm) > max_change:
#                 return False
                
#         return True

#     def calculate_heart_rate(self):
#         """Calculate heart rate from the signal buffer."""
#         if len(self.green_values) < self.min_samples:
#             self.logger.info("Not enough samples for heart rate calculation")
#             return None, 0.0

#         try:
#             # Check if it's time to update
#             current_time = time.time()
#             if current_time - self.last_update_time < self.update_interval:
#                 return self.smooth_hr, 0.0

#             signal_data = np.array(list(self.green_values))
            
#             # Remove linear trend and normalize
#             signal_data = signal.detrend(signal_data)
#             signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-6)

#             # Apply all filters
#             filtered_signal = self.apply_filters(signal_data)
            
#             # Calculate FFT
#             fft_data = np.fft.rfft(filtered_signal)
#             freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0/self.sampling_rate)
            
#             # Get the frequency range for heart rate
#             mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#             valid_freqs = freqs[mask]
#             valid_magnitude = np.abs(fft_data[mask])
            
#             if len(valid_magnitude) == 0:
#                 self.logger.warning("No valid frequencies found in signal")
#                 return self.smooth_hr, 0.0
            
#             # Normalize magnitude
#             valid_magnitude = valid_magnitude / (np.max(valid_magnitude) + 1e-6)
            
#             # Find peaks with more lenient parameters
#             peaks, properties = signal.find_peaks(valid_magnitude,
#                                                 distance=5,  # Reduced from 8
#                                                 height=0.1,  # Reduced from 0.15
#                                                 prominence=0.05)  # Reduced from 0.1
            
#             if len(peaks) == 0:
#                 self.logger.warning("No peaks found in frequency spectrum")
#                 self.no_update_count += 1
#                 if self.no_update_count > self.max_no_update:
#                     return None, 0.0
#                 return self.smooth_hr, 0.0
            
#             # Find the most prominent peak
#             peak_heights = valid_magnitude[peaks]
#             max_peak_idx = peaks[np.argmax(peak_heights)]
#             peak_freq = valid_freqs[max_peak_idx]
            
#             # Calculate heart rate
#             heart_rate = peak_freq * 60
            
#             # Validate heart rate
#             if not self.is_valid_bpm(heart_rate):
#                 self.logger.warning(f"Invalid heart rate calculated: {heart_rate}")
#                 self.no_update_count += 1
#                 if self.no_update_count > self.max_no_update:
#                     return None, 0.0
#                 return self.smooth_hr, 0.0
            
#             # Update smooth heart rate with more aggressive smoothing
#             if self.smooth_hr is None:
#                 self.smooth_hr = heart_rate
#             else:
#                 # More aggressive smoothing for stability
#                 self.smooth_hr = 0.7 * self.smooth_hr + 0.3 * heart_rate
            
#             self.no_update_count = 0
#             self.last_update_time = current_time
            
#             self.logger.info(f"Heart rate calculated: {self.smooth_hr:.1f} BPM")
#             return self.smooth_hr, 0.0
            
#         except Exception as e:
#             self.logger.error(f"Error calculating heart rate: {str(e)}")
#             return self.smooth_hr, 0.0

#     def process_frame(self, frame, roi=None):
#         """Process a single frame and update the heart rate estimate."""
#         try:
#             if frame is None or roi is None:
#                 return None, 0.0

#             # Extract the green channel average from the ROI
#             x, y, w, h = roi
#             roi_frame = frame[y:y+h, x:x+w]
#             green_value = np.mean(roi_frame[:, :, 1])
            
#             # Process the raw value
#             processed_value = self.process_value(green_value)
            
#             # Add to the signal buffer
#             self.green_values.append(processed_value)

#             # Calculate heart rate if we have enough samples
#             if len(self.green_values) >= self.min_samples:
#                 current_time = time.time()
#                 if current_time - self.last_update_time < self.update_interval:
#                     return self.smooth_hr, 1.0 if self.smooth_hr is not None else 0.0

#                 # Convert buffer to numpy array
#                 signal_data = np.array(self.green_values)
                
#                 # Remove DC component and linear trend
#                 signal_data = signal_data.astype(float)
#                 signal_data = signal_data - np.mean(signal_data)
#                 signal_data = signal.detrend(signal_data)
                
#                 # Apply bandpass filter
#                 filtered_signal = signal.filtfilt(self.b_bandpass, self.a_bandpass, signal_data)
                
#                 # Apply window function
#                 windowed_signal = filtered_signal * self.window[:len(filtered_signal)]
                
#                 # Calculate FFT
#                 fft_data = np.fft.rfft(windowed_signal)
#                 freqs = np.fft.rfftfreq(len(windowed_signal), d=1.0/self.sampling_rate)
                
#                 # Get the frequency range for heart rate
#                 mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#                 valid_freqs = freqs[mask]
#                 valid_magnitude = np.abs(fft_data[mask])
                
#                 if len(valid_magnitude) == 0:
#                     return self.smooth_hr, 0.0
                
#                 # Normalize magnitude
#                 valid_magnitude = valid_magnitude / np.max(valid_magnitude)
                
#                 # Find peaks in the magnitude spectrum
#                 peaks, properties = signal.find_peaks(valid_magnitude,
#                                                     distance=5,
#                                                     height=0.1,
#                                                     prominence=0.1)
                
#                 if len(peaks) == 0:
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
                
#                 # Find the most prominent peak
#                 peak_heights = valid_magnitude[peaks]
#                 max_peak_idx = peaks[np.argmax(peak_heights)]
#                 peak_freq = valid_freqs[max_peak_idx]
                
#                 # Calculate confidence metrics
#                 background = np.mean(valid_magnitude)
#                 snr = np.max(peak_heights) / background if background > 0 else 0
#                 peak_ratio = np.max(peak_heights) / np.sum(peak_heights)
                
#                 # Calculate heart rate
#                 heart_rate = peak_freq * 60
                
#                 # Validate heart rate
#                 if not self.is_valid_bpm(heart_rate):
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
                
#                 # Calculate confidence
#                 confidence = 0.4 * snr + 0.6 * peak_ratio
                
#                 if confidence > self.min_bpm_confidence:
#                     # Update smooth heart rate
#                     if self.smooth_hr is None:
#                         self.smooth_hr = heart_rate
#                     else:
#                         # Smooth with variable rate based on confidence
#                         alpha = min(0.7, max(0.3, confidence))
#                         self.smooth_hr = (1 - alpha) * self.smooth_hr + alpha * heart_rate
                    
#                     self.bpm_history.append(heart_rate)
#                     self.no_update_count = 0
#                     self.last_update_time = current_time
                    
#                     self.logger.info(f"Heart Rate: {self.smooth_hr:.1f} BPM (Confidence: {confidence:.2f})")
#                     return self.smooth_hr, confidence
#                 else:
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
            
#             return None, 0.0

#         except Exception as e:
#             self.logger.error(f"Error processing frame: {str(e)}")
#             return None, 0.0

#     def process_value(self, value):
#         """Process a raw green channel value."""
#         # Update baseline with slower adaptation
#         if self.baseline is None:
#             self.baseline = value
#         else:
#             # Use dynamic alpha based on signal stability
#             dynamic_alpha = min(0.95, max(0.05, 1.0 - np.std(self.values_history) if self.values_history else 0.1))
#             self.baseline = dynamic_alpha * self.baseline + (1 - dynamic_alpha) * value

#         # Store value in history
#         self.values_history.append(value)
#         if len(self.values_history) > self.window_size:
#             self.values_history.pop(0)

#         # Calculate rolling statistics
#         if len(self.values_history) >= 3:
#             recent_mean = np.mean(self.values_history[-10:] if len(self.values_history) >= 10 else self.values_history)
#             recent_std = np.std(self.values_history[-10:] if len(self.values_history) >= 10 else self.values_history)
#             # Use adaptive standard deviation threshold
#             std_dev = max(recent_std, 0.1 * abs(recent_mean - self.baseline))
#         else:
#             recent_mean = value
#             std_dev = 1.0

#         # Normalize with respect to recent statistics
#         normalized = (value - recent_mean) / (std_dev + 1e-6)  # Prevent division by zero
        
#         # Apply non-linear amplification
#         amplified = np.sign(normalized) * np.log1p(abs(normalized)) * 5.0
        
#         # Clip extreme values
#         amplified = np.clip(amplified, -10.0, 10.0)

#         self.logger.debug(f"Raw: {value:.4f}, Norm: {normalized:.4f}, Amp: {amplified:.4f}")
#         return amplified

#     def update_heart_rate(self, bpm):
#         # Add new heart rate to history
#         self.hr_history.append(bpm)
#         if len(self.hr_history) > self.hr_window:
#             self.hr_history.pop(0)

#         # Apply median filter to reject outliers
#         if len(self.hr_history) >= 3:
#             filtered_hr = np.median(self.hr_history)
#             # Only accept new heart rate if it's within 20% of median
#             if abs(bpm - filtered_hr) > 0.2 * filtered_hr:
#                 bpm = filtered_hr

#         # Smooth heart rate changes
#         if len(self.hr_history) >= 2:
#             prev_hr = self.hr_history[-2]
#             # Limit rate of change
#             max_change = 2.0  # Maximum allowed BPM change per update
#             if abs(bpm - prev_hr) > max_change:
#                 if bpm > prev_hr:
#                     bpm = prev_hr + max_change
#                 else:
#                     bpm = prev_hr - max_change

#         return bpm

#     def calculate_signal_quality(self):
#         """Calculate signal quality metrics."""
#         if len(self.green_values) < self.min_samples:
#             return 0.0
            
#         try:
#             # Convert buffer to numpy array
#             signal = np.array(self.green_values)
            
#             # Calculate signal-to-noise ratio with adaptive thresholds
#             signal_mean = np.mean(signal)
#             signal_std = np.std(signal)
            
#             # More lenient minimum threshold based on signal mean
#             min_std_threshold = 0.01 * abs(signal_mean)
#             if signal_std < min_std_threshold:
#                 return 0.0
                
#             # Calculate SNR using signal variation
#             background_noise = np.median(np.abs(np.diff(signal)))
#             snr = 20 * np.log10(signal_std / (background_noise + 1e-6))
#             snr_quality = min(1.0, max(0.0, (snr + 10) / 20))  # Adjusted scaling
            
#             # Calculate temporal consistency with windowed analysis
#             window_size = min(len(signal) // 4, 20)
#             windows = [signal[i:i+window_size] for i in range(0, len(signal)-window_size, window_size//2)]
#             if windows:
#                 window_stds = [np.std(w) for w in windows]
#                 temporal_consistency = 1.0 / (1.0 + 0.5 * np.std(window_stds) / (np.mean(window_stds) + 1e-6))
#             else:
#                 temporal_consistency = 0.0
            
#             # Calculate frequency domain metrics
#             window = signal.windows.hann(len(signal))
#             windowed = signal * window
#             fft = np.fft.rfft(windowed)
#             freqs = np.fft.rfftfreq(len(signal), d=1.0/self.sampling_rate)
            
#             # Focus on heart rate frequency range (0.7-4.0 Hz)
#             mask = (freqs >= 0.7) & (freqs <= 4.0)
#             hr_freqs = freqs[mask]
#             hr_mags = np.abs(fft[mask])
            
#             if len(hr_mags) == 0:
#                 return 0.0
                
#             # Calculate peak quality with improved metrics
#             peaks, properties = signal.find_peaks(hr_mags, distance=5, prominence=0.1)
#             if len(peaks) == 0:
#                 freq_quality = 0.0
#             else:
#                 # Calculate peak quality based on prominence and width
#                 prominences = properties["prominences"]
#                 peak_quality = np.max(prominences) / (np.sum(hr_mags) + 1e-6)
#                 freq_quality = min(1.0, peak_quality * 2.0)
            
#             self.logger.debug(f"Signal quality metrics - SNR: {snr_quality:.3f}, "
#                             f"Temporal: {temporal_consistency:.3f}, "
#                             f"Frequency: {freq_quality:.3f}")
            
#             # Combine metrics with dynamic weights based on frequency quality
#             if freq_quality > 0.5:  # Strong frequency content
#                 quality = max(0.2, (0.2 * snr_quality +
#                           0.3 * temporal_consistency +
#                           0.5 * freq_quality))  # Emphasize frequency quality
#             else:
#                 quality = max(0.2, (0.4 * snr_quality +
#                           0.4 * temporal_consistency +
#                           0.2 * freq_quality))  # More balanced weights
            
#             return quality
            
#         except Exception as e:
#             self.logger.error(f"Error calculating signal quality: {str(e)}")
#             return 0.0

#STEP 1: Face Detection & ROI Tracking (MediaPipe + Smoothing)

from collections import deque
import numpy as np
import logging
import time
from .preprocessing import SignalPreprocessor
from .ica import ICAExtractor
from .hr_estimation import HeartRateEstimator
from .filtering import HeartRateFilter, ROIStabilityChecker
from signal_processing.preprocessing import SignalPreprocessor
from signal_processing.ica import ICAExtractor
from signal_processing.hr_estimation import HeartRateEstimator
from signal_processing.facehr_estimator import FaceHREstimator
from signal_processing.filtering import HeartRateFilter, ROIStabilityChecker
from signal_processing.performance import ParallelProcessor

class SignalProcessor:
    def __init__(self, sampling_rate=30):
        self.logger = logging.getLogger(__name__)
        self.fs = sampling_rate
        self.buffer = deque(maxlen=self.fs * 10)
        self.preprocessor = SignalPreprocessor(sampling_rate)
        self.ica = ICAExtractor()
        self.hr_estimator = HeartRateEstimator(sampling_rate)
        self.facehr_estimator = FaceHREstimator(sampling_rate)  # New FaceHR estimator
        self.hr_filter = HeartRateFilter()
        self.roi_checker = ROIStabilityChecker()
        self.last_bpm = None
        self.green_values = []#TODO: check if needed
        
        # Initialization delay to avoid startup noise
        self.init_delay = 2.0  # 2 seconds delay
        self.start_time = None
        self.initialized = False
        
        # ROI tracking for automatic reset
        self.last_roi_stable = False
        self.roi_lost = False

    def process_frame(self, frame, roi):
        # Check for ROI loss first (when roi becomes None or frame becomes None)
        if (frame is None or roi is None) and self.last_roi_stable:
            print("ROI lost (None detected) - resetting signal processor")
            self.reset()
            self.roi_lost = True
            self.last_roi_stable = False
            return None, 0.0
        
        if frame is None or roi is None:
            return None, 0.0

        # Initialize start time if not set
        if self.start_time is None:
            self.start_time = time.time()
            print(f"Starting initialization delay: {self.init_delay} seconds")

        # Check if initialization delay is complete
        current_time = time.time()
        if not self.initialized and (current_time - self.start_time) < self.init_delay:
            remaining = self.init_delay - (current_time - self.start_time)
            print(f"Initialization delay: {remaining:.1f}s remaining")
            return None, 0.0
        elif not self.initialized:
            self.initialized = True
            print("Initialization complete - starting heart rate measurement")

        x, y, w, h = roi
        roi_patch = frame[y:y + h, x:x + w]

        # Check if ROI was lost and reset if needed
        current_roi_stable = self.roi_checker.is_stable(roi_patch)
        if self.last_roi_stable and not current_roi_stable:
            print("ROI lost (unstable) - resetting signal processor")
            self.reset()
            # Signal that displays should be cleared
            self.roi_lost = True
        self.last_roi_stable = current_roi_stable

        if not current_roi_stable:
            self.logger.debug("ROI patch unstable. Skipping frame.")
            return self.last_bpm, 0.0

        # Extract green channel
        green = roi_patch[:, :, 1].astype(np.float32)
        mean_val = np.mean(green)
        self.buffer.append(mean_val)

        if len(self.buffer) < self.fs * 2:
            return None, 0.0

        # Convert buffer to numpy array
        signal = np.array(self.buffer)

        # Step 1: Smooth
        signal = self.preprocessor.smooth_temporal(signal)

        # Step 2: Denoise (adaptive notch)
        signal = self.preprocessor.apply_adaptive_notch(signal)

        # Step 3: Normalize (robust)
        signal = self.preprocessor.normalize_robust(signal)

        # Step 4: Adaptive signal enhancement (replacing ICA)
        signal = self.enhance_ppg_signal(signal)

        # Step 5: Estimate heart rate using FaceHR estimator
        bpm, confidence = self.facehr_estimator.estimate(signal)

        # Step 6: Outlier rejection / temporal filtering
        filtered_bpm = self.hr_filter.update(bpm, confidence)
        self.last_bpm = filtered_bpm

        return filtered_bpm, confidence

    def enhance_ppg_signal(self, signal):
        """Enhanced PPG signal processing without ICA - more stable approach."""
        try:
            # 1. Apply bandpass filter to focus on heart rate frequencies
            from scipy.signal import butter, filtfilt
            nyquist = self.fs / 2
            low = 0.8 / nyquist  # 48 BPM
            high = 3.0 / nyquist  # 180 BPM
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # 2. Apply adaptive smoothing based on signal quality
            signal_quality = self._assess_signal_quality(filtered)
            if signal_quality > 0.7:  # High quality signal
                # Light smoothing
                from scipy.signal import savgol_filter
                smoothed = savgol_filter(filtered, 7, 2)
            else:  # Lower quality signal
                # More aggressive smoothing
                from scipy.signal import savgol_filter
                smoothed = savgol_filter(filtered, 11, 3)
            
            # 3. Remove baseline wander with more robust method
            window_size = min(30, len(smoothed) // 4)
            if window_size > 5:
                baseline = np.convolve(smoothed, np.ones(window_size)/window_size, mode='same')
                enhanced = smoothed - baseline
            else:
                enhanced = smoothed
            
            # 4. Apply temporal consistency check
            enhanced = self._apply_temporal_consistency(enhanced)
            
            print(f"Signal enhancement: quality={signal_quality:.2f}, smoothing={'light' if signal_quality > 0.7 else 'aggressive'}")
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Signal enhancement error: {e}")
            return signal
    
    def _apply_temporal_consistency(self, signal):
        """Apply temporal consistency to prevent sudden changes."""
        try:
            # Check for sudden amplitude changes
            diff = np.abs(np.diff(signal))
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            # If there are sudden large changes, apply additional smoothing
            if std_diff > 2 * mean_diff:
                from scipy.signal import savgol_filter
                signal = savgol_filter(signal, 9, 2)
                print("Applied additional smoothing due to temporal inconsistency")
            
            return signal
            
        except Exception:
            return signal
    
    def _assess_signal_quality(self, signal):
        """Assess the quality of the PPG signal."""
        try:
            # Calculate signal-to-noise ratio in heart rate band
            from scipy.signal import welch
            freqs, power = welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
            
            # Heart rate band power
            hr_mask = (freqs >= 0.8) & (freqs <= 3.0)
            hr_power = np.sum(power[hr_mask])
            total_power = np.sum(power)
            
            # SNR in heart rate band
            snr = hr_power / (total_power - hr_power + 1e-6)
            
            # Normalize to [0,1]
            quality = min(snr / 2.0, 1.0)
            
            return quality
            
        except Exception:
            return 0.5

    def reset(self):
        """Reset the signal processor state for a fresh measurement."""
        self.buffer.clear()
        self.last_bpm = None
        self.green_values.clear()
        self.start_time = None
        self.initialized = False
        
        # Reset multi-ROI buffers and qualities
        for roi_name in self.roi_buffers:
            self.roi_buffers[roi_name].clear()
            self.roi_qualities[roi_name] = 0.0
        
        # Reset FaceHR estimator
        self.facehr_estimator.reset()
        
        self.logger.debug("Signal processor reset - initialization delay will be applied")
    
    def was_roi_lost(self):
        """Check if ROI was lost and reset the flag."""
        if self.roi_lost:
            self.roi_lost = False
            return True
        return False






#STEP 2: Signal Preprocessing Enhancements




#STEP 3: ICA Optimization with PCA + Component Scoring



#STEP 4: Heart Rate Estimation Enhancements


#STEP 5: Signal Quality & Outlier Handling



#STEP 6: Performance Optimizations



#STEP 7: Testing & Evaluation Utilities

