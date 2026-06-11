"""Exercise the custom event-weighted LogisticFalkon loss."""

import sys
from pathlib import Path

import falkon
import torch
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.options import FalkonOptions

#########################################################################################################
# Import path setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nplm import EventWeightedCrossEntropyLoss


#########################################################################################################
# Encoding helpers

def encode_signed_weights(y01, active_weight):
    """Encode labels and weights in one Falkon-compatible target column.

    :param y01: Binary labels with shape ``(n_samples, 1)``.
    :param active_weight: Active event weights with shape ``(n_samples, 1)``.
    :returns: Signed target column with shape ``(n_samples, 1)``.
    """
    if y01.shape != active_weight.shape:
        raise ValueError("y01 and active_weight must have the same shape")

    return torch.where(y01 > 0.5, active_weight, -active_weight)


#########################################################################################################
# Checks

def test_constant_weights_against_builtin():
    """Compare the custom loss against Falkon's built-in weighted BCE loss.

    :returns: ``None``.
    """
    print("\n[1] Constant-weight derivative check")

    torch.manual_seed(0)

    dtype = torch.float64
    n = 1000
    neg_weight = 3.0

    kernel = falkon.kernels.GaussianKernel(1.0)

    custom_loss = EventWeightedCrossEntropyLoss(kernel)
    builtin_loss = WeightedCrossEntropyLoss(kernel, neg_weight=neg_weight)

    y01 = torch.randint(0, 2, (n, 1), dtype=dtype)
    pred = torch.randn(n, 1, dtype=dtype)

    active_weight = torch.where(
        y01 > 0.5,
        torch.ones_like(y01),
        torch.full_like(y01, neg_weight),
    )

    y_encoded = encode_signed_weights(y01, active_weight)

    checks = {
        "loss": torch.allclose(
            custom_loss(y_encoded, pred),
            builtin_loss(y01, pred),
            rtol=1e-10,
            atol=1e-10,
        ),
        "grad": torch.allclose(
            custom_loss.df(y_encoded, pred),
            builtin_loss.df(y01, pred),
            rtol=1e-10,
            atol=1e-10,
        ),
        "hess": torch.allclose(
            custom_loss.ddf(y_encoded, pred),
            builtin_loss.ddf(y01, pred),
            rtol=1e-10,
            atol=1e-10,
        ),
    }

    for name, ok in checks.items():
        print(f"{name} close: {ok}")

    assert all(checks.values())


def test_optimizer_with_synthetic_data():
    """Run a LogisticFalkon optimizer smoke test with analytic target slope.

    :returns: ``None``.
    """
    print("\n[2] LogisticFalkon optimizer smoke test")

    torch.manual_seed(1)

    dtype = torch.float64

    n_ref = 1500
    n_data = 1500

    nu = 0.5
    mu = 1.5

    x_ref = torch.randn(n_ref, 1, dtype=dtype)
    x_data = torch.randn(n_data, 1, dtype=dtype) + mu

    w_ref = torch.exp(nu * x_ref - 0.5 * nu**2)
    w_data = torch.ones(n_data, 1, dtype=dtype)

    y_ref = torch.zeros(n_ref, 1, dtype=dtype)
    y_data = torch.ones(n_data, 1, dtype=dtype)

    x_pooled = torch.cat([x_ref, x_data], dim=0)
    y01 = torch.cat([y_ref, y_data], dim=0)
    active_weight = torch.cat([w_ref, w_data], dim=0)

    active_weight = active_weight / active_weight.mean()

    assert torch.all(active_weight > 0)

    y_encoded = encode_signed_weights(y01, active_weight)

    kernel = falkon.kernels.GaussianKernel(1.5)
    loss = EventWeightedCrossEntropyLoss(kernel)

    options = FalkonOptions(use_cpu=False)

    model = falkon.LogisticFalkon(
        kernel=kernel,
        M=350,
        penalty_list=[1e-2, 1e-4, 1e-6],
        iter_list=[3, 6, 8],
        loss=loss,
        seed=123,
        options=options,
    )

    model.fit(x_pooled, y_encoded)

    grid = torch.linspace(-2.0, 4.0, 200, dtype=dtype).reshape(-1, 1)
    pred = model.predict(grid).detach().reshape(-1)

    true_f = ((mu - nu) * grid - 0.5 * (mu**2 - nu**2)).reshape(-1)

    x = grid.reshape(-1)

    x_centered = x - x.mean()
    pred_centered = pred - pred.mean()

    slope = (x_centered @ pred_centered) / (x_centered @ x_centered)
    intercept = pred.mean() - slope * x.mean()

    centered_rmse = torch.sqrt(
        torch.mean((pred_centered - (true_f - true_f.mean())) ** 2)
    )

    print(f"expected slope:       {mu - nu:.3f}")
    print(f"learned slope:        {float(slope):.3f}")
    print(f"learned intercept:    {float(intercept):.3f}")
    print(f"centered RMSE:        {float(centered_rmse):.3f}")

    assert abs(float(slope) - (mu - nu)) < 0.35


if __name__ == "__main__":
    test_constant_weights_against_builtin()
    test_optimizer_with_synthetic_data()

    print("\nAll tests passed.")
