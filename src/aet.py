"""AET/XYZ basis transforms for LISA TDI data and spectral matrices."""

from __future__ import annotations

import numpy as np


M_AET: np.ndarray = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=np.float64,
)


def xyz_to_aet_timeseries(y_xyz: np.ndarray) -> np.ndarray:
    """Transform XYZ timeseries with shape ``(N, 3)`` to AET."""
    y_xyz = np.asarray(y_xyz, dtype=np.float64)
    if y_xyz.ndim != 2 or y_xyz.shape[1] != 3:
        raise ValueError(f"Expected XYZ timeseries with shape (N, 3), got {y_xyz.shape}.")
    return (M_AET @ y_xyz.T).T


def xyz_to_aet_matrix(s_xyz: np.ndarray) -> np.ndarray:
    """Transform a spectral matrix with shape ``(..., 3, 3)`` to AET."""
    s_xyz = np.asarray(s_xyz)
    if s_xyz.shape[-2:] != (3, 3):
        raise ValueError(f"Expected XYZ matrix with trailing shape (3, 3), got {s_xyz.shape}.")
    matrix = M_AET.astype(s_xyz.dtype, copy=False)
    return matrix @ s_xyz @ matrix.conj().T


def aet_to_xyz_matrix(s_aet: np.ndarray) -> np.ndarray:
    """Transform a spectral matrix with shape ``(..., 3, 3)`` from AET to XYZ."""
    s_aet = np.asarray(s_aet)
    if s_aet.shape[-2:] != (3, 3):
        raise ValueError(f"Expected AET matrix with trailing shape (3, 3), got {s_aet.shape}.")
    matrix_t = M_AET.T.astype(s_aet.dtype, copy=False)
    return matrix_t @ s_aet @ matrix_t.conj().T
