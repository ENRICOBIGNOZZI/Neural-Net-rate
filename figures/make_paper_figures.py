#!/usr/bin/env python3
"""Generate three paper figures for the neural feature-learning rate project.

Figures
-------
1. fig_rate_regimes.{png,pdf}
   Theoretical rate exponent alpha_NN(b,q) versus the learning-clock exponent q,
   together with fixed-kernel (Caponnetto--De Vito) benchmarks.
2. fig_training_dynamics.{png,pdf}
   Full-batch population-proxy training dynamics for a frozen random-feature model,
   a trainable shallow MLP, and a trainable residual MLP: loss R_t, q_t, and
   effective dimension N_t(lambda).
3. fig_spectral_tube.{png,pdf}
   Empirical spectral trajectories for shallow MLPs at several widths together
   with the finite-width tube based on the initial spectrum.

The numerical experiment is deliberately small and reproducible. It is meant as a
sanity check of the theory rather than a benchmark study.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)


def alpha_nn(b: float, q: np.ndarray) -> np.ndarray:
    return (b * q) / ((b + 1.0) * q + 1.0)


def alpha_kernel(b0: float, r: float) -> float:
    return (b0 * r) / (b0 * r + 1.0)


def plot_rate_regimes(outdir: Path) -> None:
    q = np.linspace(0.05, 8.0, 500)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for b in (1.25, 2.0, 4.0):
        ax.plot(q, alpha_nn(b, q), lw=2.0, label=rf"NN: $b={b:g}$")

    for r, ls in ((0.5, "--"), (1.0, ":")):
        a = alpha_kernel(2.0, r)
        ax.axhline(a, lw=1.6, ls=ls, label=rf"Kernel: $b_0=2,\ r={r:g}$")

    ax.set_xlabel(r"Learning-clock exponent $q$")
    ax.set_ylabel(r"Statistical exponent $\alpha$ in $T^{-\alpha}$")
    ax.set_title("Rate regimes: learned geometry versus fixed-kernel benchmark")
    ax.set_xlim(q.min(), q.max())
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_rate_regimes.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_population_grid(n: int = 512, d: int = 2, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = 2.0 * torch.rand((n, d), generator=gen) - 1.0
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1.0)
    x = x / norm
    y = (
        torch.sin(3.0 * x[:, 0])
        + 0.65 * torch.cos(4.0 * x[:, 1])
        + 0.40 * torch.sin(2.5 * (x[:, 0] + x[:, 1]))
        + 0.25 * x[:, 0] * x[:, 1]
    ).unsqueeze(1)
    y = y - y.mean()
    return x, y


class ShallowMLP(nn.Module):
    def __init__(self, d: int, m: int, train_hidden: bool = True, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m = m
        self.W = nn.Parameter(torch.randn((m, d), generator=g) / math.sqrt(d), requires_grad=train_hidden)
        self.a = nn.Parameter(torch.zeros((m, 1)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.W.T) / math.sqrt(self.m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self.a


class ResidualMLP(nn.Module):
    def __init__(self, d: int, m: int, depth: int = 6, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m = m
        self.depth = depth
        self.tau = 1.0 / math.sqrt(depth)
        self.W_in = nn.Parameter(torch.randn((m, d), generator=g) / math.sqrt(d))
        self.blocks = nn.ParameterList([
            nn.Parameter(torch.randn((m, m), generator=g) / math.sqrt(m))
            for _ in range(depth)
        ])
        self.a = nn.Parameter(torch.zeros((m, 1)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x @ self.W_in.T)
        for W in self.blocks:
            h = h + self.tau * torch.tanh(h @ W.T)
        return h / math.sqrt(self.m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self.a


@dataclass
class Trace:
    t: list
    loss: list
    q: list
    neff: list
    eigvals: list
    beta: list


def feature_spectrum(model: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        phi = model.features(x)
        G = (phi.T @ phi) / x.shape[0]
        eig = torch.linalg.eigvalsh(G).flip(0).clamp_min(0.0)
    return eig


def train_trace(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    steps: int,
    lam: float,
    record_every: int,
) -> Trace:
    trace = Trace([], [], [], [], [], [])
    params = [p for p in model.parameters() if p.requires_grad]

    pred = model(x)
    loss0 = 0.5 * torch.mean((pred - y) ** 2)
    R0 = float(loss0.detach())

    for step in range(steps + 1):
        pred = model(x)
        loss = 0.5 * torch.mean((pred - y) ** 2)

        grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
        grad_sq = sum(float(torch.sum(g.detach() ** 2)) for g in grads)
        t = max((step + 1) * lr, lr)
        q = t * grad_sq / max(float(loss.detach()), 1e-16)

        if step % record_every == 0 or step == steps:
            eig = feature_spectrum(model, x)
            neff = float(torch.sum(eig / (eig + lam)))
            R = float(loss.detach())
            beta = math.sqrt(max(t * (R0 - R), 0.0) / model.m)
            trace.t.append(t)
            trace.loss.append(R)
            trace.q.append(q)
            trace.neff.append(neff)
            trace.eigvals.append(eig.cpu().numpy())
            trace.beta.append(beta)

        if step == steps:
            break
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.add_(g, alpha=-lr)

    return trace


def plot_training_dynamics(outdir: Path, fast: bool = False) -> None:
    x, y = make_population_grid(n=384 if fast else 768, seed=11)
    m = 96 if fast else 160
    steps = 450 if fast else 1100
    lr = 0.025
    rec = 5 if fast else 10
    lam = 0.01

    models = {
        "Frozen random features": ShallowMLP(2, m, train_hidden=False, seed=4),
        "Shallow MLP": ShallowMLP(2, m, train_hidden=True, seed=4),
        "Residual MLP": ResidualMLP(2, m, depth=6, seed=4),
    }
    traces = {name: train_trace(model, x, y, lr, steps, lam, rec) for name, model in models.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for name, tr in traces.items():
        axes[0].plot(tr.t, tr.loss, lw=1.8, label=name)
        axes[1].plot(tr.t, tr.q, lw=1.8, label=name)
        axes[2].plot(tr.t, tr.neff, lw=1.8, label=name)

    axes[0].set_yscale("log")
    axes[0].set_title(r"Population-proxy risk $\mathcal{R}_t$")
    axes[0].set_xlabel("Gradient-flow time (discretized)")
    axes[0].set_ylabel("Risk")

    axes[1].set_title(r"Learning clock $q_t$")
    axes[1].set_xlabel("Gradient-flow time (discretized)")
    axes[1].set_ylabel(r"$q_t \approx t\|\nabla R_t\|^2/R_t$")

    axes[2].set_title(r"Effective dimension $\mathcal{N}_t(\lambda)$")
    axes[2].set_xlabel("Gradient-flow time (discretized)")
    axes[2].set_ylabel(rf"$\mathcal{{N}}_t({lam:g})$")

    for ax in axes:
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Kernel-like versus feature-learning dynamics", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_training_dynamics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_tube(outdir: Path, fast: bool = False) -> None:
    x, y = make_population_grid(n=320 if fast else 640, seed=21)
    widths = (48, 96, 192) if fast else (64, 160, 384)
    steps = 320 if fast else 850
    lr = 0.02
    rec = 8 if fast else 10
    lam = 0.01
    modes = 3

    fig, axes = plt.subplots(1, len(widths), figsize=(13.0, 3.6), sharey=False)
    for ax, m in zip(axes, widths):
        model = ShallowMLP(2, m, train_hidden=True, seed=8)
        tr = train_trace(model, x, y, lr, steps, lam, rec)
        eig0 = tr.eigvals[0]
        t = np.asarray(tr.t)
        beta = np.asarray(tr.beta)

        for j in range(min(modes, len(eig0))):
            mu = np.asarray([e[j] for e in tr.eigvals])
            root0 = math.sqrt(max(eig0[j], 0.0))
            lo = np.maximum(root0 - beta, 0.0) ** 2
            hi = (root0 + beta) ** 2
            ax.plot(t, mu, lw=1.7, label=rf"$\mu_{{{j+1}}}(t)$")
            ax.fill_between(t, lo, hi, alpha=0.12)

        ax.set_title(rf"Width $m={m}$")
        ax.set_xlabel("Gradient-flow time")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Leading feature-covariance eigenvalues")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Finite-width spectral motion and theoretical tubes", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_spectral_tube.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("figures/output"))
    parser.add_argument("--fast", action="store_true", help="Use smaller experiments for CI/smoke testing.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)
    plot_rate_regimes(args.outdir)
    plot_training_dynamics(args.outdir, fast=args.fast)
    plot_spectral_tube(args.outdir, fast=args.fast)
    print(f"Wrote figures to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
