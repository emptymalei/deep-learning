# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "onnx>=1.16",
#     "onnxruntime>=1.18",
#     "onnxscript>=0.2",
#     "torch>=2.6",
# ]
# ///
"""Train a small pendulum forecaster and export it to ONNX for the browser.

Run from the repository root:

    uv run scripts/pendulum_onnx.py

Writes

* ``dl/assets/models/pendulum-mlp.onnx``: the model, opset 17, input ``context``
  with shape ``[batch, k]`` and output ``forecast`` with shape ``[batch, h]``,
  angles in radians. The normalisation is part of the graph.
* ``dl/assets/models/pendulum-mlp.json``: the simulation settings the browser
  needs to reproduce the data, plus validation metrics.

The widget ``dl/assets/widgets/pendulum-onnx.js`` reads both files, so if you
change the settings here, re-run the script and the widget follows.

The data follows the pendulum chapter of the book
(``dl/time-series-deep-learning/timeseries.dataset.pendulum.md``), with two
changes: we integrate the full nonlinear equation instead of using the
small-angle solution, and we vary the length and the initial angle so the
model has to read them off the context window.

This script lives outside ``dl/`` on purpose: the site build turns the ``.py``
notebooks under ``dl/`` into pages.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "dl" / "assets" / "models"
ONNX_PATH = OUT_DIR / "pendulum-mlp.onnx"
META_PATH = OUT_DIR / "pendulum-mlp.json"

# ---------------------------------------------------------------------------
# Physics, matched to the book's pendulum chapter.
#
# The book uses theta(t) = theta0 cos(2 pi t / p) exp(-beta t) with L = 100 m,
# g = 9.81, beta = 0.001 and 400 samples per period. The equation of motion
# with that envelope is
#
#     theta'' = -(g / L) sin(theta) - 2 beta theta'
#
# (with sin(theta) ~ theta it gives back the book's formula). We integrate it
# with RK4 at the book's step, and the model sees every 10th step, which is the
# same 10-step gap the feedforward chapter uses.
# ---------------------------------------------------------------------------
G = 9.81
BOOK_LENGTH = 100.0
BETA = 0.001
RK4_DT = 2 * math.pi * math.sqrt(BOOK_LENGTH / G) / 400  # ~0.0502 s
SUBSTEPS = 10
DT = RK4_DT * SUBSTEPS  # ~0.502 s between the samples the model sees

LENGTH_RANGE = (50.0, 150.0)  # metres, around the book's 100 m
THETA0_RANGE_DEG = (5.0, 170.0)

# Forecasting task.
K = 24  # context length (samples)
H = 24  # forecast horizon (samples)
AUTOREGRESSIVE_ROUNDS = 3  # the widget's rollout mode: 3 x H steps

# Data and training.
SEED = 7
N_TRAIN_TRAJ = 2400
N_VAL_TRAJ = 400
TRAJ_SAMPLES = 160  # 80 s per trajectory
HIDDEN_WIDTHS = [256, 128, 64, 128, 256]  # the book's [512, 256, 64, 256, 512], halved
EPOCHS = 40
BATCH_SIZE = 1024
LR = 3e-3


def simulate(
    theta0: np.ndarray,
    length: np.ndarray,
    n_samples: int,
    g: float = G,
    beta: float = BETA,
    dt: float = RK4_DT,
    substeps: int = SUBSTEPS,
) -> np.ndarray:
    """Integrate many pendulums at once with RK4, starting from rest.

    Returns angles in radians with shape ``(len(theta0), n_samples)``;
    sample 0 is ``theta0`` and consecutive samples are ``substeps`` RK4 steps
    apart. The widget's JavaScript port must stay step-for-step identical.
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    omega = np.zeros_like(theta)
    w2 = g / np.asarray(length, dtype=np.float64)

    def accel(th, om):
        return -w2 * np.sin(th) - 2.0 * beta * om

    out = np.empty((theta.size, n_samples), dtype=np.float64)
    out[:, 0] = theta
    for i in range(1, n_samples):
        for _ in range(substeps):
            k1t, k1w = omega, accel(theta, omega)
            k2t = omega + 0.5 * dt * k1w
            k2w = accel(theta + 0.5 * dt * k1t, k2t)
            k3t = omega + 0.5 * dt * k2w
            k3w = accel(theta + 0.5 * dt * k2t, k3t)
            k4t = omega + dt * k3w
            k4w = accel(theta + dt * k3t, k4t)
            theta = theta + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
            omega = omega + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        out[:, i] = theta
    return out


def sample_trajectories(n: int, rng: np.random.Generator) -> np.ndarray:
    theta0 = np.deg2rad(rng.uniform(*THETA0_RANGE_DEG, size=n))
    length = rng.uniform(*LENGTH_RANGE, size=n)
    return simulate(theta0, length, TRAJ_SAMPLES)


def make_windows(
    series: np.ndarray, k: int = K, h: int = H
) -> tuple[np.ndarray, np.ndarray]:
    """All (context, target) windows at every offset of every trajectory."""
    span = k + h
    win = np.lib.stride_tricks.sliding_window_view(series, span, axis=1)
    win = win.reshape(-1, span)
    return win[:, :k].astype(np.float32), win[:, k:].astype(np.float32)


# ---------------------------------------------------------------------------
# Model: the feedforward network from the book's feedforward chapter, with the
# input scaling and a "last value + correction" output baked in, so the
# exported graph takes and returns plain angles in radians.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TSFFNParams:
    hidden_widths: list[int]


class TSFeedForward(nn.Module):
    """Feedforward network for univariate time series (as in the book)."""

    def __init__(self, history_length: int, horizon: int, ffn_params: TSFFNParams):
        super().__init__()
        self.history_length = history_length
        self.horizon = horizon
        widths = ffn_params.hidden_widths
        self.regulate_input = nn.Linear(history_length, widths[0])
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())
                for d_in, d_out in zip(widths[:-1], widths[1:])
            ]
        )
        self.regulate_output = nn.Linear(widths[-1], horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regulate_output(self.hidden_layers(self.regulate_input(x)))


class PendulumForecaster(nn.Module):
    """Angles in, angles out, with the normalisation inside the graph.

    Each window is divided by its own amplitude (the largest |angle| in the
    context), so a 5 degree swing looks like a 170 degree one to the MLP. The
    amplitude also goes in as one extra input, because a pendulum that swings
    wider also swings more slowly. The MLP predicts the change from the last
    context value, in units of that amplitude.
    """

    def __init__(self, ffn: TSFeedForward, scale: float, eps: float = 1e-3):
        super().__init__()
        self.ffn = ffn
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def amplitude(self, context: torch.Tensor) -> torch.Tensor:
        return context.abs().amax(dim=1, keepdim=True) + self.eps

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        last = context[:, -1:]
        amp = self.amplitude(context)
        features = torch.cat([context / amp, amp / self.scale], dim=1)
        return last + amp * self.ffn(features)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    )


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [
            model(torch.from_numpy(x[i : i + batch])).numpy()
            for i in range(0, len(x), batch)
        ]
    )


@torch.no_grad()
def rollout(model: nn.Module, context: np.ndarray, rounds: int) -> np.ndarray:
    """Autoregressive forecast: feed each forecast back in as context."""
    ctx = context.copy()
    preds = []
    for _ in range(rounds):
        f = predict(model, ctx)
        preds.append(f)
        ctx = np.concatenate([ctx, f], axis=1)[:, -K:]
    return np.concatenate(preds, axis=1)


def train() -> tuple[PendulumForecaster, dict]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    t0 = time.perf_counter()
    train_series = sample_trajectories(N_TRAIN_TRAJ, rng)
    val_series = sample_trajectories(N_VAL_TRAJ, rng)
    x_train, y_train = make_windows(train_series)
    # theta -> -theta is a symmetry of the pendulum: use it as free data.
    x_train = np.concatenate([x_train, -x_train])
    y_train = np.concatenate([y_train, -y_train])
    x_val, y_val = make_windows(val_series)
    print(
        f"data: {len(x_train):,} train / {len(x_val):,} val windows "
        f"from {N_TRAIN_TRAJ} / {N_VAL_TRAJ} trajectories "
        f"({time.perf_counter() - t0:.1f} s)"
    )

    scale = float(x_train.std())
    ffn = TSFeedForward(K + 1, H, TSFFNParams(hidden_widths=HIDDEN_WIDTHS))
    model = PendulumForecaster(ffn, scale)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {K}+1 -> {HIDDEN_WIDTHS} -> {H}, {n_params:,} parameters")

    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    steps_per_epoch = math.ceil(len(xt) / BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch, pct_start=0.15
    )

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), BATCH_SIZE):
            xb, yb = xt[perm[i : i + BATCH_SIZE]], yt[perm[i : i + BATCH_SIZE]]
            # Error relative to each window's amplitude, so small swings count
            # as much as large ones.
            err = (model(xb) - yb) / model.amplitude(xb)
            loss = err.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            val = rmse(predict(model, x_val), y_val)
            print(
                f"epoch {epoch + 1:3d}  train rel. mse {total / len(xt):.2e}  "
                f"val rmse {val:.5f} rad  ({time.perf_counter() - t0:.0f} s)"
            )
    train_seconds = time.perf_counter() - t0

    # Metrics, all in radians.
    naive_val = np.repeat(x_val[:, -1:], H, axis=1)
    # What the widget shows: context = the first K samples, starting from rest.
    rest_x = val_series[:, :K].astype(np.float32)
    rest_y = val_series[:, K : K + H]
    rest_y3 = val_series[:, K : K + AUTOREGRESSIVE_ROUNDS * H]
    metrics = {
        "val_rmse_rad": rmse(predict(model, x_val), y_val),
        "val_naive_rmse_rad": rmse(naive_val, y_val),
        "val_from_rest_rmse_rad": rmse(predict(model, rest_x), rest_y),
        "val_from_rest_naive_rmse_rad": rmse(
            np.repeat(rest_x[:, -1:], H, axis=1), rest_y
        ),
        "val_autoregressive_rmse_rad": rmse(
            rollout(model, rest_x, AUTOREGRESSIVE_ROUNDS), rest_y3
        ),
        "val_autoregressive_naive_rmse_rad": rmse(
            np.repeat(rest_x[:, -1:], AUTOREGRESSIVE_ROUNDS * H, axis=1), rest_y3
        ),
    }
    info = {
        "n_params": n_params,
        "scale": scale,
        "train_seconds": round(train_seconds, 1),
        "n_train_windows": len(x_train),
        "n_val_windows": len(x_val),
        "x_val_sample": x_val[:512],
    }
    return model, metrics | info


def export(model: PendulumForecaster, x_check: np.ndarray) -> float:
    """Export to ONNX (opset 17) and check onnxruntime against torch."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.from_numpy(x_check[:2])
    try:
        program = torch.onnx.export(
            model,
            (example,),
            dynamo=True,
            opset_version=17,
            input_names=["context"],
            output_names=["forecast"],
            dynamic_shapes={"context": {0: torch.export.Dim("batch")}},
            external_data=False,
            verbose=False,
        )
        program.save(str(ONNX_PATH), external_data=False)
    except Exception as err:  # older/newer torch: fall back to the TorchScript exporter
        print(f"dynamo export failed ({err!r}), using the TorchScript exporter")
        torch.onnx.export(
            model,
            (example,),
            str(ONNX_PATH),
            dynamo=False,
            opset_version=17,
            input_names=["context"],
            output_names=["forecast"],
            dynamic_axes={"context": {0: "batch"}, "forecast": {0: "batch"}},
        )

    # Rename the batch dimension and drop any doc strings with local paths.
    m = onnx.load(str(ONNX_PATH))
    for value in (*m.graph.input, *m.graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    m.doc_string = ""
    m.producer_name = "pytorch"
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, str(ONNX_PATH))

    opset = {o.domain: o.version for o in m.opset_import}.get("", None)
    assert opset == 17, f"expected opset 17, got {opset}"
    assert [i.name for i in m.graph.input] == ["context"], m.graph.input
    assert [o.name for o in m.graph.output] == ["forecast"], m.graph.output
    in_dims = [
        d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim
    ]
    out_dims = [
        d.dim_param or d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim
    ]
    assert in_dims == ["batch", K] and out_dims == ["batch", H], (in_dims, out_dims)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    (got,) = sess.run(["forecast"], {"context": x_check})
    want = predict(model, x_check)
    max_diff = float(np.max(np.abs(got - want)))
    print(
        f"onnx: opset {opset}, {in_dims} -> {out_dims}, max |onnx - torch| = {max_diff:.2e}"
    )
    assert max_diff < 1e-4, max_diff
    return max_diff


def main() -> None:
    model, result = train()
    max_diff = export(model, result.pop("x_val_sample"))

    deg = 180 / math.pi
    meta = {
        "name": "pendulum-mlp",
        "description": (
            "Feedforward network that maps the last k pendulum angles to the next h "
            "angles (radians). Trained offline with PyTorch by scripts/pendulum_onnx.py."
        ),
        "input": {
            "name": "context",
            "shape": ["batch", K],
            "dtype": "float32",
            "units": "rad",
        },
        "output": {
            "name": "forecast",
            "shape": ["batch", H],
            "dtype": "float32",
            "units": "rad",
        },
        "opset": 17,
        "k": K,
        "h": H,
        "autoregressive_rounds": AUTOREGRESSIVE_ROUNDS,
        "dt": DT,
        "rk4_dt": RK4_DT,
        "rk4_substeps": SUBSTEPS,
        "g": G,
        "L_range": list(LENGTH_RANGE),
        "theta0_range_deg": list(THETA0_RANGE_DEG),
        "damping": BETA,
        "equation": "theta'' = -(g/L) sin(theta) - 2 * damping * theta', from rest at theta0",
        "architecture": {
            "hidden_widths": HIDDEN_WIDTHS,
            "activation": "relu",
            "n_params": result["n_params"],
            "amplitude": "a = max|context| + 1e-3",
            "input_scale": result["scale"],
            "mlp_input": "[context / a, a / input_scale] (k + 1 values)",
            "output": "last context value + a * mlp(...)",
        },
        "metrics": {
            "val_rmse_rad": result["val_rmse_rad"],
            "val_naive_rmse_rad": result["val_naive_rmse_rad"],
            "val_rmse_deg": result["val_rmse_rad"] * deg,
            "val_naive_rmse_deg": result["val_naive_rmse_rad"] * deg,
            "val_from_rest_rmse_rad": result["val_from_rest_rmse_rad"],
            "val_from_rest_naive_rmse_rad": result["val_from_rest_naive_rmse_rad"],
            "val_autoregressive_rmse_rad": result["val_autoregressive_rmse_rad"],
            "val_autoregressive_naive_rmse_rad": result[
                "val_autoregressive_naive_rmse_rad"
            ],
            "notes": (
                "val_*: all windows of held-out trajectories; from_rest: context is the first k "
                "samples (what the widget shows); autoregressive: 3 x h steps, forecasts fed back."
            ),
        },
        "training": {
            "seed": SEED,
            "train_trajectories": N_TRAIN_TRAJ,
            "val_trajectories": N_VAL_TRAJ,
            "samples_per_trajectory": TRAJ_SAMPLES,
            "train_windows": result["n_train_windows"],
            "val_windows": result["n_val_windows"],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seconds": result["train_seconds"],
            "torch": torch.__version__,
        },
        "onnx_max_abs_diff_vs_torch": max_diff,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")

    m = meta["metrics"]
    print(
        f"val RMSE {m['val_rmse_deg']:.3f} deg vs naive {m['val_naive_rmse_deg']:.2f} deg | "
        f"from rest {m['val_from_rest_rmse_rad'] * deg:.3f} deg | "
        f"autoregressive 3h {m['val_autoregressive_rmse_rad'] * deg:.3f} deg"
    )
    for path in (ONNX_PATH, META_PATH):
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
