"""Rigid geometry shared by feature estimation and pixel registration."""

import numpy as np

from .shared import FloatArray


def fit_rigid_transform(source: FloatArray, target: FloatArray) -> FloatArray:
    """Kabsch fit from source to target, with unit scale and no reflection."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = target_mean - rotation @ source_mean
    return transform
