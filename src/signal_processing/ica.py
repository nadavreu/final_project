import numpy as np
from sklearn.decomposition import FastICA
from scipy.ndimage import shift
from scipy.stats import skew

class ICAExtractor:
    def __init__(self, n_components=5, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def extract_best_component(self, signal):
        try:
            # Create phase-shifted matrix using scipy.ndimage.shift
            shifts = [0, 1, 2, 3, 4]
            X = np.column_stack([shift(signal, s, mode='nearest') for s in shifts])

            # Apply PCA to reduce noise
            pca = PCA(n_components=self.n_components)
            X_pca = pca.fit_transform(X)

            # Apply ICA
            ica = FastICA(n_components=self.n_components, random_state=self.random_state, max_iter=1000)
            S = ica.fit_transform(X_pca)

            # Score components using periodicity, skewness, and SNR
            scores = [self._score_component(comp) for comp in S.T]
            best_idx = np.argmax(scores)

            return S[:, best_idx]
        except Exception as e:
            print(f"ICA extraction error: {e}")
            return signal

    def _score_component(self, comp):
        periodicity = self._estimate_periodicity(comp)
        skewness = abs(skew(comp))
        snr = np.std(comp) / (np.mean(np.abs(np.diff(comp))) + 1e-6)
        return 0.4 * periodicity + 0.3 * skewness + 0.3 * snr

    def _estimate_periodicity(self, comp):
        corr = np.correlate(comp, comp, mode='full')
        corr = corr[len(corr)//2:]
        peaks = np.diff(np.sign(np.diff(corr))) < 0
        peak_indices = np.where(peaks)[0]
        if len(peak_indices) < 2:
            return 0
        intervals = np.diff(peak_indices)
        return 1.0 / (np.std(intervals) + 1e-6)