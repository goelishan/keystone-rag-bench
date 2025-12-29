import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors row-wise.

    Args:
        vectors: np.ndarray of shape (N, D)

    Returns:
        np.ndarray of shape (N, D), L2-normalized

    Raises:
        ValueError: if vectors contain NaNs, Infs, or zero-norm rows
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError("vectors must be a numpy ndarray")

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D array (N, D), got shape {vectors.shape}"
        )

    if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
        raise ValueError("NaN or infinite values found in vectors")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Zero-norm vector(s) found; cannot normalize")

    return vectors / norms
