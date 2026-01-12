from .base import AnomalyDetector, ensure_2d
import numpy as np
from scipy.stats import zscore

class EnsembleAverage(AnomalyDetector):
    """Ensemble that averages normalized scores from multiple base detectors."""
    
    def __init__(self, base_detectors=None):
        """
        Args:
            base_detectors: List of instantiated detector objects (not factories).
        """
        self.base_detectors = base_detectors or []
        self.fitted_detectors = []

    def fit(self, X, y=None):
        X = ensure_2d(X)
        self.fitted_detectors = []
        for detector in self.base_detectors:
            detector.fit(X, y)
            self.fitted_detectors.append(detector)
        return self

    def score(self, X):
        X = ensure_2d(X)
        if not self.fitted_detectors:
            raise ValueError("No fitted detectors available.")
        
        all_scores = []
        for detector in self.fitted_detectors:
            s = detector.score(X)
            # Normalize to z-scores for fair averaging
            s_norm = zscore(s)
            # Handle constant scores (nan from zscore)
            s_norm = np.nan_to_num(s_norm, nan=0.0)
            all_scores.append(s_norm)
        
        # Average across all detectors
        return np.mean(all_scores, axis=0)
