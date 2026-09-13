# Neural-Net-rate

Reproducible figures for the finite-width neural feature-learning rate project.

The current repository contains three paper-oriented figures:

1. **Rate regimes** — compares the feature-learning exponent
   \(\alpha_{\mathrm{NN}}=bq/((b+1)q+1)\) with fixed-kernel / Caponnetto–De Vito benchmarks.
2. **Training dynamics** — compares a frozen random-feature model, a trainable shallow MLP, and a residual MLP through population-proxy risk \(\mathcal R_t\), learning clock \(q_t\), and effective dimension \(\mathcal N_t(\lambda)\).
3. **Finite-width spectral tube** — tracks leading eigenvalues of the learned feature covariance at several widths and overlays the finite-width tube predicted by the theory.

The numerical experiments are deliberately small synthetic sanity checks. They are **not** intended as benchmark evidence; their purpose is to visualize the quantities that enter the theoretical results.

## Reproduce

```bash
python -m pip install -r requirements.txt
python figures/make_paper_figures.py
```

For a fast smoke test:

```bash
python figures/make_paper_figures.py --fast
```

Outputs are written to `figures/output/` in both PNG and PDF format.

## Main theoretical quantities visualized

The feature-learning rate exponent used in Figure 1 is

\[
\alpha_{\mathrm{NN}}(b,q)
=
\frac{bq}{(b+1)q+1},
\]

while a fixed-kernel benchmark with spectral exponent \(b_0\) and source exponent \(r\) has

\[
\alpha_{\mathrm K}(b_0,r)
=
\frac{b_0r}{b_0r+1}.
\]

The learning clock is estimated from discretized full-batch gradient flow as

\[
q_t \approx t\,\frac{\|\nabla \mathcal R_t\|^2}{\mathcal R_t},
\]

and the effective dimension is

\[
\mathcal N_t(\lambda)
=
\sum_j \frac{\mu_j(t)}{\mu_j(t)+\lambda}.
\]

The spectral-tube plot uses the finite-width control

\[
\left|\sqrt{\mu_j(t)}-\sqrt{\mu_j(0)}\right|
\lesssim
\sqrt{\frac{t(\mathcal R_0-\mathcal R_t)}{m}}
\]

for the bounded-input, 1-Lipschitz `tanh` experiment.
