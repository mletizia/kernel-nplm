# event_weighted_cross_entropy.py

import torch
import torch.nn.functional as F

from falkon.gsc_losses import Loss
from falkon.options import FalkonOptions
import falkon
from typing import Optional


class EventWeightedCrossEntropyLoss(Loss):
    r"""
    Per-event weighted binary cross entropy for LogisticFalkon.

    The target tensor Y must have shape (n, 1) and encodes both
    the class label and the active event weight:

        Y_i < 0  -> class 0, active weight |Y_i|
        Y_i > 0  -> class 1, active weight |Y_i|

    Loss:
        l_i = w_i * [ y_i * softplus(-f_i)
                    + (1 - y_i) * softplus(f_i) ]

    This is equivalent to

        w0_i * (1-y_i) * log(1 + exp(f_i))
      + w1_i * y_i     * log(1 + exp(-f_i))

    with only the active class weight stored for each event.
    """

    def __init__(
        self,
        kernel: falkon.kernels.Kernel,
        opt: Optional[FalkonOptions] = None,
    ):
        super().__init__(
            name="EventWeightedCrossEntropy",
            kernel=kernel,
            opt=opt,
        )

    @staticmethod
    def _split_target(encoded_y: torch.Tensor):
        # encoded_y has shape (n, 1)
        # class label: 0 for negative/reference, 1 for positive/data
        y = (encoded_y > 0).to(dtype=encoded_y.dtype)

        # active event weight
        w = encoded_y.abs()

        return y, w

    def __call__(self, encoded_y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        y, w = self._split_target(encoded_y)

        return w * (
            y * F.softplus(-pred)
            + (1.0 - y) * F.softplus(pred)
        )

    def df(self, encoded_y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""
        First derivative with respect to pred.

        d l / d f =
            w * [ (1-y) * sigmoid(f) - y * sigmoid(-f) ]
        """
        y, w = self._split_target(encoded_y)

        sig = torch.sigmoid(pred)

        return w * (
            (1.0 - y) * sig
            - y * (1.0 - sig)
        )

    def ddf(self, encoded_y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""
        Second derivative with respect to pred.

        d^2 l / d f^2 =
            w * sigmoid(f) * sigmoid(-f)

        This remains positive if all event weights are non-negative.
        """
        _, w = self._split_target(encoded_y)

        sig = torch.sigmoid(pred)

        return w * sig * (1.0 - sig)

    def __repr__(self):
        return f"EventWeightedCrossEntropy(kernel={self.kernel})"