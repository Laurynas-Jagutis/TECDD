import numpy as np
from sklearn.decomposition import PCA

PCA_VARIANCE = 0.95
THRESHOLD_PERCENTILE = 99.5      # detection threshold theta
UPDATE_PERCENTILE = 75           # update threshold theta'
REFERENCE_SIZE = 400             # points in a reference sample
KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0)
SEED = 0


def _pairwise_sq_dists(A, B):
    """Squared Euclidean distances between every row of A and every row of B"""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


class DriftDetector:
    """PCA projection, RBF kernel MMD and the frozen threshold of TECDD"""
    
    def __init__(self):
        self.pca = PCA(n_components=PCA_VARIANCE)
        self.threshold = None
        self.reference_update_threshold = None
        self.reference_sample = None
        self.bandwidths = None
        
    def transform(self, embeddings):
        return self.pca.transform(embeddings).astype(np.float32, copy=False)
    
    def _sample_reference(self, reduced_windows):
        """Pool the points(embeddings after PCA) of several windwows and sample REFERENCE_SIZE of them"""
        pooled = np.concatenate([np.asarray(window, dtype=np.float32) for window in reduced_windows if np.asarray(window).size > 0],
                                axis=0)
        if pooled.shape[0] > REFERENCE_SIZE:
            idx = np.random.default_rng(SEED).choice(pooled.shape[0], REFERENCE_SIZE, replace=False)
            pooled = pooled[np.sort(idx)]
        return pooled
    
    @staticmethod
    def _median_bandwidth(points):
        """Median distance between pairs of points of the reference sample."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[0] > 500:
            idx = np.random.default_rng(SEED + 1).choice(pts.shape[0], 500, replace=False)
            pts = pts[idx]
        d2 = _pairwise_sq_dists(pts, pts)
        upper = np.triu_indices(pts.shape[0], k=1)
        if upper[0].size == 0:
            return 1.0
        median = float(np.median(np.sqrt(d2[upper])))
        return median if median > 1e-6 else 1e-6

    def mmd2(self, X, Y):
        """Biased MMD^2 (Eq. 1), with the kernel averaged over the three bandwidths (Eq. 2)."""
        dxx = _pairwise_sq_dists(X, X)
        dyy = _pairwise_sq_dists(Y, Y)
        dxy = _pairwise_sq_dists(X, Y)
        total = 0.0
        for sigma in self.bandwidths:
            gamma = 1.0 / (2.0 * sigma * sigma)
            total += (
                np.exp(-gamma * dxx).mean()
                + np.exp(-gamma * dyy).mean()
                - 2.0 * np.exp(-gamma * dxy).mean()
            )
        return max(float(total / len(self.bandwidths)), 0.0)
    
    def distance(self, windows_embeddings, reference_sample):
        """MMD^2 between a window of embeddings and a reference sample"""
        return self.mmd2(self.transform(windows_embeddings), reference_sample)
    
    def fit(self, opening_embeddings, opening_window_embeddings):
        """Stage 2: fit the PCA, R0, h, theta and theta' on the opening region."""
        self.pca.fit(opening_embeddings)
        print(f"PCA reduced dimensionality: {self.pca.n_components_}")
        reduced_windows = [self.transform(w) for w in opening_window_embeddings if w.size > 0]
        self.reference_sample = self._sample_reference(reduced_windows)
        h = self._median_bandwidth(self.reference_sample)
        self.bandwidths = [m * h for m in KERNEL_MULTIPLIERS]
        distances = np.array(
            [self.mmd2(window, self.reference_sample) for window in reduced_windows], dtype=np.float32
        )
        self.threshold = float(np.percentile(distances, THRESHOLD_PERCENTILE))
        self.reference_update_threshold = float(np.percentile(distances, UPDATE_PERCENTILE))
        print(
            f"R0: {self.reference_sample.shape[0]} points, h={h:.4f}, "
            f"theta={self.threshold:.6f}, theta'={self.reference_update_threshold:.6f}"
        )
    
    def calibrate_reference_bank(self, window_embeddings):
        """Pool a bank of windows into a reference sample and score each window against it"""
        reduced_windows = [self.transform(w) for w in window_embeddings if w.size > 0]
        reference_sample = self._sample_reference(reduced_windows)
        distances = np.array([self.mmd2(window, reference_sample) for window in reduced_windows], dtype=np.float32)
        
        return {
            "reference_sample": reference_sample,
            "threshold": float(np.percentile(distances, THRESHOLD_PERCENTILE)),
            "distances": distances,
        }
        
    def export_state(self):
        return{
            "pca_components": PCA_VARIANCE,
            "pca_mean": np.asarray(self.pca.mean_, dtype=np.float32).tolist(),
            "pca_components_matrix": np.asarray(self.pca.components_, dtype=np.float32).tolist(),
            "pca_explained_variance": np.asarray(self.pca.explained_variance_, dtype=np.float32).tolist(),
            "threshold": float(self.threshold),
            "reference_update_threshold": float(self.reference_update_threshold),
            "mmd_reference_sample": np.asarray(self.reference_sample, dtype=np.float32).tolist(),
            "mmd_bandwidths": [float(b) for b in self.bandwidths],
        }
    
    @classmethod
    def from_state(cls, state):
        """Rebuilding the frozen detector from export_state() without fitting anything"""
        detector = cls()
        components = np.asarray(state["pca_components_matrix"], dtype=np.float32)
        pca = PCA(n_components=components.shape[0])
        pca.components_ = components
        pca.mean_ = np.asarray(state["pca_mean"], dtype=np.float32)
        pca.n_components_ = components.shape[0]
        pca.n_features_in_ = components.shape[1]
        pca.explained_variance_ = np.asarray(state["pca_explained_variance"], dtype=np.float32)
        detector.pca = pca
        detector.threshold = float(state["threshold"])
        detector.reference_update_threshold = float(state["reference_update_threshold"])
        detector.reference_sample = np.asarray(state["mmd_reference_sample"], dtype=np.float32)
        detector.bandwidths = [float(b) for b in state["mmd_bandwidths"]]
        return detector
