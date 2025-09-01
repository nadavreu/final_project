"""
Multi-ROI Fusion Module for PulseVision

This module implements quality-weighted fusion of multiple ROI signals
to improve heart rate estimation accuracy and robustness.
"""

import numpy as np
import logging
from scipy import signal
from scipy.stats import pearsonr
from collections import deque


class ROIFusion:
    """
    Quality-weighted fusion of multiple ROI signals for improved heart rate estimation.
    
    This class implements several fusion strategies:
    1. Quality-weighted averaging
    2. Correlation-based selection
    3. Adaptive fusion based on signal characteristics
    """
    
    def __init__(self, min_correlation=0.3, quality_threshold=0.2):
        """
        Initialize the ROI fusion system.
        
        Args:
            min_correlation (float): Minimum correlation threshold for signal fusion
            quality_threshold (float): Minimum quality threshold for ROI inclusion
        """
        self.logger = logging.getLogger(__name__)
        self.min_correlation = min_correlation
        self.quality_threshold = quality_threshold
        
        # Fusion history for adaptive strategies
        self.fusion_history = deque(maxlen=10)
        self.quality_history = {
            'forehead': deque(maxlen=10),
            'left_cheek': deque(maxlen=10),
            'right_cheek': deque(maxlen=10)
        }
        
        # Fusion weights (will be updated based on performance)
        self.adaptive_weights = {
            'forehead': 0.4,
            'left_cheek': 0.3,
            'right_cheek': 0.3
        }
        
        self.logger.info("ROI Fusion initialized with correlation threshold: %.2f", 
                        self.min_correlation)

    def fuse_signals(self, roi_signals, roi_qualities):
        """
        Fuse multiple ROI signals using quality-weighted approach.
        
        Args:
            roi_signals (dict): Dictionary of ROI signals {roi_name: signal_array}
            roi_qualities (dict): Dictionary of ROI qualities {roi_name: quality_score}
            
        Returns:
            tuple: (fused_signal, fusion_confidence)
        """
        try:
            if not roi_signals or len(roi_signals) < 2:
                self.logger.warning("Insufficient ROI signals for fusion")
                return None, 0.0
            
            # Filter ROIs by quality threshold
            valid_rois = self._filter_by_quality(roi_signals, roi_qualities)
            if len(valid_rois) < 2:
                self.logger.warning("Insufficient high-quality ROIs for fusion")
                return self._fallback_fusion(roi_signals, roi_qualities)
            
            # Align signals to same length
            aligned_signals = self._align_signals(valid_rois)
            if aligned_signals is None:
                return self._fallback_fusion(roi_signals, roi_qualities)
            
            # Calculate correlation matrix
            correlations = self._calculate_correlations(aligned_signals)
            
            # Select fusion strategy based on signal characteristics
            fusion_strategy = self._select_fusion_strategy(correlations, roi_qualities)
            
            if fusion_strategy == "quality_weighted":
                fused_signal, confidence = self._quality_weighted_fusion(
                    aligned_signals, roi_qualities
                )
            elif fusion_strategy == "correlation_based":
                fused_signal, confidence = self._correlation_based_fusion(
                    aligned_signals, correlations, roi_qualities
                )
            else:  # adaptive
                fused_signal, confidence = self._adaptive_fusion(
                    aligned_signals, correlations, roi_qualities
                )
            
            # Update fusion history for adaptive learning
            self._update_fusion_history(fused_signal, confidence, roi_qualities)
            
            self.logger.debug(f"Fused {len(valid_rois)} ROIs using {fusion_strategy} strategy, "
                            f"confidence: {confidence:.3f}")
            
            return fused_signal, confidence
            
        except Exception as e:
            self.logger.error(f"Error in signal fusion: {e}")
            return self._fallback_fusion(roi_signals, roi_qualities)

    def _filter_by_quality(self, roi_signals, roi_qualities):
        """Filter ROIs by quality threshold."""
        valid_rois = {}
        for roi_name, signal in roi_signals.items():
            if roi_qualities.get(roi_name, 0) >= self.quality_threshold:
                valid_rois[roi_name] = signal
        return valid_rois

    def _align_signals(self, roi_signals):
        """Align all signals to the same length (shortest signal)."""
        try:
            min_length = min(len(signal) for signal in roi_signals.values())
            if min_length < 10:  # Minimum signal length
                return None
                
            aligned = {}
            for roi_name, signal in roi_signals.items():
                aligned[roi_name] = signal[-min_length:]  # Take last min_length samples
            return aligned
        except Exception:
            return None

    def _calculate_correlations(self, aligned_signals):
        """Calculate pairwise correlations between ROI signals."""
        roi_names = list(aligned_signals.keys())
        n_rois = len(roi_names)
        correlations = np.zeros((n_rois, n_rois))
        
        for i, roi1 in enumerate(roi_names):
            for j, roi2 in enumerate(roi_names):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    try:
                        corr, _ = pearsonr(aligned_signals[roi1], aligned_signals[roi2])
                        correlations[i, j] = max(0, corr)  # Only positive correlations
                    except:
                        correlations[i, j] = 0.0
        
        return correlations

    def _select_fusion_strategy(self, correlations, roi_qualities):
        """Select the best fusion strategy based on signal characteristics."""
        # Calculate average correlation
        n_rois = len(correlations)
        if n_rois < 2:
            return "quality_weighted"
            
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_tri = correlations[np.triu_indices(n_rois, k=1)]
        avg_correlation = np.mean(upper_tri)
        
        # Calculate quality spread
        qualities = list(roi_qualities.values())
        quality_spread = np.std(qualities) if len(qualities) > 1 else 0
        
        # Strategy selection logic
        if avg_correlation > 0.7 and quality_spread < 0.2:
            return "quality_weighted"  # High correlation, similar quality
        elif avg_correlation < 0.4:
            return "correlation_based"  # Low correlation, use best correlated pairs
        else:
            return "adaptive"  # Mixed conditions, use adaptive approach

    def _quality_weighted_fusion(self, aligned_signals, roi_qualities):
        """Quality-weighted fusion of ROI signals."""
        try:
            roi_names = list(aligned_signals.keys())
            signals = np.array([aligned_signals[name] for name in roi_names])
            qualities = np.array([roi_qualities[name] for name in roi_names])
            
            # Normalize qualities to sum to 1
            weights = qualities / np.sum(qualities)
            
            # Weighted average
            fused_signal = np.average(signals, axis=0, weights=weights)
            
            # Confidence based on quality consistency
            quality_consistency = 1.0 - np.std(qualities) / (np.mean(qualities) + 1e-6)
            confidence = np.mean(qualities) * quality_consistency
            
            return fused_signal, min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in quality-weighted fusion: {e}")
            return None, 0.0

    def _correlation_based_fusion(self, aligned_signals, correlations, roi_qualities):
        """Correlation-based fusion selecting best signal pairs."""
        try:
            roi_names = list(aligned_signals.keys())
            n_rois = len(roi_names)
            
            # Find the pair with highest correlation
            best_corr = 0
            best_pair = None
            
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    if correlations[i, j] > best_corr:
                        best_corr = correlations[i, j]
                        best_pair = (i, j)
            
            if best_pair is None or best_corr < self.min_correlation:
                # Fallback to quality-weighted
                return self._quality_weighted_fusion(aligned_signals, roi_qualities)
            
            # Fuse the best pair
            roi1, roi2 = best_pair
            signal1 = aligned_signals[roi_names[roi1]]
            signal2 = aligned_signals[roi_names[roi2]]
            quality1 = roi_qualities[roi_names[roi1]]
            quality2 = roi_qualities[roi_names[roi2]]
            
            # Weight by quality
            total_quality = quality1 + quality2
            weight1 = quality1 / total_quality
            weight2 = quality2 / total_quality
            
            fused_signal = weight1 * signal1 + weight2 * signal2
            
            # Confidence based on correlation and quality
            confidence = best_corr * (total_quality / 2.0)
            
            return fused_signal, min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in correlation-based fusion: {e}")
            return self._quality_weighted_fusion(aligned_signals, roi_qualities)

    def _adaptive_fusion(self, aligned_signals, correlations, roi_qualities):
        """Adaptive fusion combining multiple strategies."""
        try:
            # Get quality-weighted result
            qw_signal, qw_confidence = self._quality_weighted_fusion(aligned_signals, roi_qualities)
            
            # Get correlation-based result
            cb_signal, cb_confidence = self._correlation_based_fusion(aligned_signals, correlations, roi_qualities)
            
            if qw_signal is None or cb_signal is None:
                return qw_signal, qw_confidence if qw_signal is not None else (cb_signal, cb_confidence)
            
            # Adaptive weighting based on historical performance
            qw_weight = self.adaptive_weights.get('quality_weighted', 0.5)
            cb_weight = 1.0 - qw_weight
            
            # Combine the two approaches
            fused_signal = qw_weight * qw_signal + cb_weight * cb_signal
            
            # Combined confidence
            confidence = qw_weight * qw_confidence + cb_weight * cb_confidence
            
            return fused_signal, min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in adaptive fusion: {e}")
            return self._quality_weighted_fusion(aligned_signals, roi_qualities)

    def _fallback_fusion(self, roi_signals, roi_qualities):
        """Fallback fusion when primary methods fail."""
        try:
            if not roi_signals:
                return None, 0.0
            
            # Use the highest quality ROI
            best_roi = max(roi_qualities.items(), key=lambda x: x[1])
            best_signal = roi_signals[best_roi[0]]
            best_quality = best_roi[1]
            
            self.logger.debug(f"Using fallback fusion with {best_roi[0]} (quality: {best_quality:.3f})")
            
            return best_signal, best_quality * 0.8  # Reduce confidence for fallback
            
        except Exception as e:
            self.logger.error(f"Error in fallback fusion: {e}")
            return None, 0.0

    def _update_fusion_history(self, fused_signal, confidence, roi_qualities):
        """Update fusion history for adaptive learning."""
        try:
            # Store fusion result
            self.fusion_history.append({
                'signal': fused_signal.copy() if fused_signal is not None else None,
                'confidence': confidence,
                'timestamp': len(self.fusion_history)
            })
            
            # Update quality history for each ROI
            for roi_name, quality in roi_qualities.items():
                if roi_name in self.quality_history:
                    self.quality_history[roi_name].append(quality)
            
            # Update adaptive weights based on recent performance
            self._update_adaptive_weights()
            
        except Exception as e:
            self.logger.error(f"Error updating fusion history: {e}")

    def _update_adaptive_weights(self):
        """Update adaptive weights based on recent fusion performance."""
        try:
            if len(self.fusion_history) < 5:
                return  # Need more history
            
            # Analyze recent performance
            recent_confidences = [entry['confidence'] for entry in list(self.fusion_history)[-5:]]
            avg_confidence = np.mean(recent_confidences)
            
            # Adjust weights based on performance
            if avg_confidence > 0.7:
                # Good performance, maintain current weights
                pass
            elif avg_confidence < 0.4:
                # Poor performance, increase quality-weighted weight
                self.adaptive_weights['quality_weighted'] = min(0.8, 
                    self.adaptive_weights.get('quality_weighted', 0.5) + 0.1)
            
            self.logger.debug(f"Updated adaptive weights: {self.adaptive_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating adaptive weights: {e}")

    def get_fusion_statistics(self):
        """Get statistics about fusion performance."""
        try:
            if not self.fusion_history:
                return {}
            
            confidences = [entry['confidence'] for entry in self.fusion_history]
            
            stats = {
                'avg_confidence': np.mean(confidences),
                'max_confidence': np.max(confidences),
                'min_confidence': np.min(confidences),
                'fusion_count': len(self.fusion_history),
                'adaptive_weights': self.adaptive_weights.copy()
            }
            
            # ROI quality statistics
            for roi_name, quality_history in self.quality_history.items():
                if quality_history:
                    stats[f'{roi_name}_avg_quality'] = np.mean(quality_history)
                    stats[f'{roi_name}_quality_trend'] = np.mean(list(quality_history)[-3:]) - np.mean(list(quality_history)[:3])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting fusion statistics: {e}")
            return {}

    def reset(self):
        """Reset fusion history and adaptive weights."""
        self.fusion_history.clear()
        for roi_name in self.quality_history:
            self.quality_history[roi_name].clear()
        
        # Reset to default weights
        self.adaptive_weights = {
            'forehead': 0.4,
            'left_cheek': 0.3,
            'right_cheek': 0.3
        }
        
        self.logger.debug("ROI fusion system reset")
