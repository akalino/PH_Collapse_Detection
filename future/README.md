# Expansion Roadmap

- Scale up ambient dimension and sample size, run calibration pipeline
- Track collapse dynamically on snapshots along controlled collapse trajectories

## Idea

Two objectives:

### Objective 1

See if the mechanism taxonomy stats consistent with the original results.
Questions:

- Does DTM continue to improve robustness relative to VR?
- Is MTE still more sensitive than TP arcoss mechanisms?
- Does calibration remain near nominal over the expanded null suite?
- Does power improve or degrade in a predictive fashion w.r.t $n, d, \varepsilon$?

### Objective 2

Instead of estimating only on the `bookends', we can sample along the collapse trajectory

$X^{(0)}, X^{(1)}, \ldots, X^{(T)}$

where $X^{(0)}$ is healthy and $X^{(T)}$ is the final collapsed state.
Questions:

- How early in the process does the PH summary detect collapse?
- Which summaries respond monotonically (or near monotonically)?
- Do the PH summaries fire before the standard spectral metrics?

## Experiments

### Scaling 

We keep the same mechanisms as the original paper:
- Mechanism A: linear/spectral collapse,
- Mechanism B: nonlinear support collapse, and
- Mechanism C: contamination.

We use the same PH filtrations:
- VR
- DTM
- Consider witness complex again for speed/robustness comparison.

We can expand the PH metrics:
- Total persistence (TP)
- Mean tail excess (MTE)
- Max persistence (MP)
- Top five persistence (TFP)
- Betti curve area (BCA)
- Betti curve peak (BCP)
- Betti curve delta (BCD)

still computed over homology dimensions $q \in \{0,1,2\}.

Expansions to point cloud sizes:
- Initial aim:
-- $n \in \{250, 500, 1000, 2000, 5000\}$
-- $d \in \{25, 50, 100, 200, 300\}$
-- stress test at $n=10000, d \in \{200, 500\}

For each mechanism class and null, start with 100 replications, then progress to 500 for finalization.

For each alternative collapse condition $\theta$, estimate the power
Compute a regression on performance as a function

$\widehat{\pi}(\theta) \sim f(n,d,\varepsilon,\text{mechanism, filtration, metrics})$

### Snapshot / trajectory study

For each geometry family, define a trajectory

$$
X^{(t)} = \Phi_t(X^{(0)}), \qquad t=0,1,\dots,T,
$$

where $\Phi_0$ is the identity and $\Phi_T$ produces the target collapsed configuration.

#### Linear collapse

Let $X^{(0)} \subset \mathbb{R}^d$ be healthy. Define

$$
X^{(t)} = A_t X^{(0)},
$$

where $A_t$ gradually suppresses variance along selected directions.

For example,

$$
A_t = \mathrm{diag}(1,\dots,1,\lambda_t,\dots,\lambda_t),
\qquad
\lambda_t \downarrow 0.
$$

#### Nonlinear-support collapse

Map points toward a lower-dimensional nonlinear set $M$ by

$$
X^{(t)} = (1-\gamma_t)X^{(0)} + \gamma_t \,\Pi_M(X^{(0)}),
$$

where $\Pi_M$ is a projection or nearest-point map onto $M$, and

$$
0=\gamma_0 < \gamma_1 < \cdots < \gamma_T = 1.
$$

#### Contamination / heterogeneity collapse

Increase contamination or mixture imbalance over time:

$$
X^{(t)} \sim (1-\rho_t)P_{\mathrm{healthy}} + \rho_t P_{\mathrm{contam}},
\qquad
\rho_t \uparrow 1.
$$

### Detection-time metrics

Define the first detection time of a test statistic $T$ as

$$
\tau_{\mathrm{det}}(T)
=
\min\{t : T(X^{(t)}) > c_\alpha(T;N)\},
$$

with $\tau_{\mathrm{det}}(T)=\infty$ if no detection occurs.

To compare methods, summarize:

$$
\mathbb{E}[\tau_{\mathrm{det}}(T)],
\qquad
\Pr\big(\tau_{\mathrm{det}}(T) < \infty\big),
$$

and the distribution of $\tau_{\mathrm{det}}(T)$ across replicates.

A useful normalized version is

$$
\widetilde{\tau}_{\mathrm{det}}(T)
=
\frac{\tau_{\mathrm{det}}(T)}{T},
$$

so values lie in $[0,1]$.

### Monotonicity and smoothness

To understand whether a statistic behaves predictably during collapse, measure whether $T_t$ is approximately monotone in $t$.

Possible diagnostics:

- Spearman correlation between $t$ and $T_t$,
- number of sign changes in first differences,
- total variation

$$
\mathrm{TV}(T_\bullet)=\sum_{t=1}^{T} |T_t - T_{t-1}|.
$$

This helps distinguish stable early-warning signals from noisy endpoint-only detectors.
