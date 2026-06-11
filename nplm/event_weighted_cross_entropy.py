"""Define a per-event weighted cross-entropy loss for LogisticFalkon."""

import torch
import torch.nn.functional as F

from falkon.gsc_losses import Loss


#########################################################################################################
# Public loss class

class EventWeightedCrossEntropyLoss(Loss):
    r"""Per-event weighted binary cross entropy for LogisticFalkon.

    The target tensor encodes both the class label and the active event weight:

    ``Y_i < 0`` maps to class 0 with active weight ``abs(Y_i)``.
    ``Y_i > 0`` maps to class 1 with active weight ``abs(Y_i)``.

    :param kernel: Falkon kernel used by the loss.
    :param opt: Optional Falkon options.
    """

    def __init__(
        self,
        kernel,
        opt=None,
    ):
        """Initialize the Falkon loss object.

        :param kernel: Falkon kernel used by the loss.
        :param opt: Optional Falkon options.
        """
        super().__init__(
            name="EventWeightedCrossEntropy",
            kernel=kernel,
            opt=opt,
        )

    @staticmethod
    def _split_target(encoded_y):
        """Split encoded targets into binary labels and active weights.

        :param encoded_y: Signed targets with shape ``(n_samples, 1)``.
        :returns: Pair ``(labels, weights)`` with shape ``(n_samples, 1)``.
        """
        if encoded_y.ndim != 2 or encoded_y.shape[1] != 1:
            raise ValueError("encoded_y must have shape (n_samples, 1)")

        y = (encoded_y > 0).to(dtype=encoded_y.dtype)
        w = encoded_y.abs()

        return y, w

    def __call__(self, encoded_y, pred):
        """Evaluate the pointwise loss.

        :param encoded_y: Signed targets with shape ``(n_samples, 1)``.
        :param pred: Model predictions with shape ``(n_samples, 1)``.
        :returns: Pointwise loss values with shape ``(n_samples, 1)``.
        """
        y, w = self._split_target(encoded_y)

        return w * (
            y * F.softplus(-pred)
            + (1.0 - y) * F.softplus(pred)
        )

    def df(self, encoded_y, pred):
        r"""Return the first derivative with respect to the prediction.

        :param encoded_y: Signed targets with shape ``(n_samples, 1)``.
        :param pred: Model predictions with shape ``(n_samples, 1)``.
        :returns: Pointwise first derivatives with shape ``(n_samples, 1)``.
        """
        y, w = self._split_target(encoded_y)

        sig = torch.sigmoid(pred)

        return w * (
            (1.0 - y) * sig
            - y * (1.0 - sig)
        )

    def ddf(self, encoded_y, pred):
        r"""Return the second derivative with respect to the prediction.

        :param encoded_y: Signed targets with shape ``(n_samples, 1)``.
        :param pred: Model predictions with shape ``(n_samples, 1)``.
        :returns: Pointwise second derivatives with shape ``(n_samples, 1)``.
        """
        _, w = self._split_target(encoded_y)

        sig = torch.sigmoid(pred)

        return w * sig * (1.0 - sig)

    def __repr__(self):
        """Return a compact representation of the loss.

        :returns: Human-readable representation string.
        """
        return f"EventWeightedCrossEntropy(kernel={self.kernel})"
