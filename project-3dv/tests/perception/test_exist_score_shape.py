"""
test_exist_score_shape.py
=========================
Regression test: SuperdecFitter.fit_adaptive() must not crash when the
model outputs exist scores with a trailing dimension.

Background
----------
A specific deployment checkpoint returns out['exist'] with shape (1, N, 1)
instead of the expected (1, N).  fit_adaptive() then does:

    exist_score = out['exist'][0].cpu().numpy()   # (N, 1) — bad
    ...
    if float(exist_score[p_idx]) < self.exist_threshold:   # TypeError!

exist_score[p_idx] is shape (1,), not a 0-d scalar, so float() raises:
    TypeError: only 0-dimensional arrays can be converted to Python scalars

Fix: exist_score = out['exist'][0].cpu().numpy().ravel()   # always (N,)

To verify the fix is necessary:
    1. Remove the .ravel() call in superdec_fitter.py line ~672.
    2. Run this file — test_exist_score_trailing_dim should FAIL with TypeError.
    3. Restore .ravel() — all tests should PASS.

No checkpoint or GPU required: self.model is replaced with a stub.
"""

import numpy as np
import pytest
import torch

from superdec_fitter import SuperdecFitter
from superquadric import MultiSQFit

# Number of candidate primitives SuperDec always outputs.
_N_PRIMS = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fitter(exist_trailing_dim: bool) -> SuperdecFitter:
    """
    Build a SuperdecFitter without loading a real checkpoint.

    Parameters
    ----------
    exist_trailing_dim : bool
        True  → out['exist'] has shape (1, N, 1) — the buggy deployment shape.
        False → out['exist'] has shape (1, N)   — the expected shape.
    """
    fitter = object.__new__(SuperdecFitter)
    fitter.device          = "cpu"
    fitter.exist_threshold = 0.3
    fitter.n_points        = 4096

    def _normalize(pts: np.ndarray):
        # Return pts unchanged; zero translation; scale 1.
        return pts, np.zeros(3, dtype=np.float64), 1.0

    def _denormalize(out: dict, t_batch, s_batch) -> dict:
        return out  # identity — values already in "world" units

    def _model_stub(x: torch.Tensor) -> dict:
        B = x.shape[0]  # always 1
        # exist scores: all 1.0 so every primitive exceeds exist_threshold
        # and float(exist_score[p_idx]) is actually exercised.
        if exist_trailing_dim:
            exist = torch.ones(B, _N_PRIMS, 1)   # (1, 16, 1) — buggy shape
        else:
            exist = torch.ones(B, _N_PRIMS)      # (1, 16)   — correct shape

        return {
            "exist":  exist,
            "scale":  torch.full((B, _N_PRIMS, 3), 0.05),
            "rotate": torch.eye(3).unsqueeze(0).unsqueeze(0)
                          .expand(B, _N_PRIMS, 3, 3).contiguous(),
            "trans":  torch.zeros(B, _N_PRIMS, 3),
            "shape":  torch.ones(B, _N_PRIMS, 2),   # e1=e2=1.0 → sphere
        }

    fitter.model                = _model_stub
    fitter._normalize_points    = _normalize
    fitter._denormalize_outdict = _denormalize
    return fitter


def _point_cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    """
    Random cloud satisfying _check_input_contract:
        shape[0] >= 100, |coords| <= 2, std > 0.01
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trailing_dim", [False, True],
                         ids=["shape_N", "shape_N_1"])
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_exist_score_shape(trailing_dim: bool) -> None:
    """
    fit_adaptive() must return a MultiSQFit for both out['exist'] shapes.

    With trailing_dim=True and the .ravel() fix absent this raises on older
    NumPy:
        TypeError: only 0-dimensional arrays can be converted to Python scalars
    and on NumPy ≥1.25 emits a DeprecationWarning (promoted to an error here
    via filterwarnings so the test fails correctly on all NumPy versions).
    Restore the fix to make both cases pass.
    """
    fitter = _make_fitter(exist_trailing_dim=trailing_dim)
    pts    = _point_cloud()
    result = fitter.fit_adaptive(pts)
    assert isinstance(result, MultiSQFit), (
        f"Expected MultiSQFit, got {type(result)}"
    )


def test_exist_score_primitives_returned() -> None:
    """
    With all exist scores = 1.0 (above threshold 0.3), at least one
    primitive must survive filtering and appear in the result.
    """
    fitter = _make_fitter(exist_trailing_dim=True)
    pts    = _point_cloud()
    result = fitter.fit_adaptive(pts)
    assert len(result.primitives) > 0, (
        "Expected at least one primitive when all exist scores are 1.0"
    )


def test_exist_score_shape_conf_is_scalar() -> None:
    """
    shape_conf on each returned primitive must be a plain Python float,
    not an array — verifies the second float(exist_score[p_idx]) call
    (used to populate shape_conf) is also fixed.
    """
    fitter = _make_fitter(exist_trailing_dim=True)
    pts    = _point_cloud()
    result = fitter.fit_adaptive(pts)
    for prim in result.primitives:
        assert isinstance(prim.shape_conf, float), (
            f"shape_conf should be float, got {type(prim.shape_conf)}"
        )
