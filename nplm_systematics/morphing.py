"""Cached nuisance morphing arrays for profiled likelihood evaluations."""

import json
from pathlib import Path

import numpy as np


#########################################################################################################
# Validation helpers

def _as_2d_delta(delta, name):
    """Convert cached response values to a two-dimensional float array.

    :param delta: Response array with shape ``(n_events,)`` or ``(n_events, n_nuisance)``.
    :param name: Name used in validation errors.
    :returns: Contiguous response array with shape ``(n_events, n_nuisance)``.
    """
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_events,) or (n_events, n_nuisance)")
    return np.ascontiguousarray(arr)


#########################################################################################################
# Cached linear morphing

class LinearLogRMorphingCache:
    """Cached linear nuisance response values.

    The cache represents ``log r(x; nu) = sum_a nu_a * delta_a(x)`` on fixed
    reference and data/event arrays.
    """

    def __init__(self, delta_ref, delta_data=None, clip=30.0, metadata=None):
        """Initialize the morphing cache.

        :param delta_ref: Reference response with shape ``(n_ref,)`` or ``(n_ref, n_nuisance)``.
        :param delta_data: Optional data response with shape ``(n_data,)`` or ``(n_data, n_nuisance)``.
        :param clip: Absolute clipping threshold applied before exponentiation.
        :param metadata: Optional metadata dictionary.
        """
        self.delta_ref = _as_2d_delta(delta_ref, "delta_ref")
        self.delta_data = None
        if delta_data is not None:
            self.delta_data = _as_2d_delta(delta_data, "delta_data")
            if self.delta_data.shape[1] != self.delta_ref.shape[1]:
                raise ValueError("delta_ref and delta_data must have the same number of nuisances")

        self.clip = float(clip)
        if self.clip <= 0 or not np.isfinite(self.clip):
            raise ValueError("clip must be positive and finite")

        self.metadata = dict(metadata or {})

    @property
    def n_nuisance(self):
        """Number of nuisance directions represented by the cache.

        :returns: Number of nuisance directions.
        """
        return int(self.delta_ref.shape[1])

    def _nu_array(self, nu):
        """Convert nuisance input to a validated one-dimensional array.

        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Nuisance array with shape ``(n_nuisance,)``.
        """
        arr = np.asarray(nu, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("nu must be a scalar or one-dimensional array")
        if arr.shape[0] != self.n_nuisance:
            raise ValueError(f"Expected {self.n_nuisance} nuisance values, got {arr.shape[0]}")
        return arr

    def _log_r(self, delta, nu):
        """Evaluate and clip cached log-ratios.

        :param delta: Response array with shape ``(n_events, n_nuisance)``.
        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Clipped log-ratio with shape ``(n_events,)``.
        """
        log_r = delta @ self._nu_array(nu)
        return np.clip(log_r, -self.clip, self.clip)

    def log_r_ref(self, nu):
        """Evaluate ``log r`` on cached reference events.

        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Reference log-ratio with shape ``(n_ref,)``.
        """
        return self._log_r(self.delta_ref, nu)

    def r_ref(self, nu):
        """Evaluate ``r`` on cached reference events.

        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Reference ratio with shape ``(n_ref,)``.
        """
        return np.exp(self.log_r_ref(nu))

    def log_r_data(self, nu):
        """Evaluate ``log r`` on cached data events.

        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Data log-ratio with shape ``(n_data,)``.
        """
        if self.delta_data is None:
            raise RuntimeError("This cache does not contain delta_data")
        return self._log_r(self.delta_data, nu)

    def r_data(self, nu):
        """Evaluate ``r`` on cached data events.

        :param nu: Scalar nuisance value or array with shape ``(n_nuisance,)``.
        :returns: Data ratio with shape ``(n_data,)``.
        """
        return np.exp(self.log_r_data(nu))

    def save_npz(self, path):
        """Save cached response arrays to a compressed NumPy file.

        :param path: Output ``.npz`` path.
        :returns: ``None``.
        """
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
    def load_npz(cls, path):
        """Load a cache saved with :meth:`save_npz`.

        :param path: Input ``.npz`` path.
        :returns: ``LinearLogRMorphingCache`` instance.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            delta_data = data["delta_data"] if "delta_data" in data else None
            return cls(
                delta_ref=data["delta_ref"],
                delta_data=delta_data,
                clip=float(data["clip"]),
                metadata=metadata,
            )
