"""Cached nuisance morphing arrays for profiled likelihood evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np


def _as_2d_delta(delta: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_events,) or (n_events, n_nuisance)")
    return np.ascontiguousarray(arr)


@dataclass
class LinearLogRMorphingCache:
    """Cached linear nuisance response values.

    The cache represents

    ``log r(x; nu) = sum_a nu_a * delta_a(x)``

    on fixed reference and data/event arrays. This avoids serializing Falkon
    models when the profiled statistic only needs repeated evaluations on the
    same events.
    """

    delta_ref: np.ndarray
    delta_data: Optional[np.ndarray] = None
    clip: float = 30.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta_ref = _as_2d_delta(self.delta_ref, "delta_ref")
        if self.delta_data is not None:
            self.delta_data = _as_2d_delta(self.delta_data, "delta_data")
            if self.delta_data.shape[1] != self.delta_ref.shape[1]:
                raise ValueError("delta_ref and delta_data must have the same number of nuisances")

        self.clip = float(self.clip)
        if self.clip <= 0 or not np.isfinite(self.clip):
            raise ValueError("clip must be positive and finite")

    @property
    def n_nuisance(self) -> int:
        """Number of nuisance directions represented by the cache."""
        return int(self.delta_ref.shape[1])

    def _nu_array(self, nu: np.ndarray) -> np.ndarray:
        arr = np.asarray(nu, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("nu must be a scalar or one-dimensional array")
        if arr.shape[0] != self.n_nuisance:
            raise ValueError(f"Expected {self.n_nuisance} nuisance values, got {arr.shape[0]}")
        return arr

    def _log_r(self, delta: np.ndarray, nu: np.ndarray) -> np.ndarray:
        log_r = delta @ self._nu_array(nu)
        return np.clip(log_r, -self.clip, self.clip)

    def log_r_ref(self, nu: np.ndarray) -> np.ndarray:
        """Evaluate ``log r`` on cached reference events."""
        return self._log_r(self.delta_ref, nu)

    def r_ref(self, nu: np.ndarray) -> np.ndarray:
        """Evaluate ``r`` on cached reference events."""
        return np.exp(self.log_r_ref(nu))

    def log_r_data(self, nu: np.ndarray) -> np.ndarray:
        """Evaluate ``log r`` on cached data events."""
        if self.delta_data is None:
            raise RuntimeError("This cache does not contain delta_data")
        return self._log_r(self.delta_data, nu)

    def r_data(self, nu: np.ndarray) -> np.ndarray:
        """Evaluate ``r`` on cached data events."""
        return np.exp(self.log_r_data(nu))

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save cached response arrays to a compressed NumPy file."""
        path = Path(path)
        payload = {
            "delta_ref": self.delta_ref,
            "clip": np.asarray(self.clip, dtype=np.float64),
            "metadata_json": np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        }
        if self.delta_data is not None:
            payload["delta_data"] = self.delta_data
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "LinearLogRMorphingCache":
        """Load a cache saved with :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            delta_data = data["delta_data"] if "delta_data" in data else None
            return cls(
                delta_ref=data["delta_ref"],
                delta_data=delta_data,
                clip=float(data["clip"]),
                metadata=metadata,
            )
