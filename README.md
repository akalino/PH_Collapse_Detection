# WIP: PH Collapse Detection

#### Goal
Use persistent homology to detect degenerating covariance / 
collapse to lower intrinsic dimension.

At some juncture in studying the topology of point clouds, there becomes 
less interest in studying the actual topology, and the question turns to:

Can we detect, quantify, or test that collapse is happening using 
persistence?

#### Basic setup

$X_i = Y_i + \epsilon Z_i$ where $Y_i \sim \mu$ is supported on some
$k$-manifold $\mathcal{M} \subset \mathbb{R}^d$ and 
$Z_i \sim \mathcal{N}(0, \Sigma)$ with $\epsilon \rightarrow 0$.
Here, $\epsilon$ is the coefficient for added isotropic noise
(big is more noise, small is less noise).  
**So larger $\epsilon$ mean a further drift away from the underlying collapse.**

To consider: 
* [ ] include $\epsilon=0$ as a noise-free manifold? (immediate complex co/dis-nnect)

We're interested in when the covariance matrix $\Sigma_\epsilon$ 
collapses ($\rightarrow \inf$),
or **equivalently when the effective dimension drops from $d$ to $k$**
w.r.t $\epsilon$ increases.

Questions then become:
* can we test a hypothesis of "full dimension vs. collapsed dimension"?
* can we use barcodes to estimate the intrinsic dimension?


#### Claim to experiment

Persistence-based statistics can consistently 
distinguish $H_1$ from $H_0$ as anisotropy increases, even when
classical covariance-based tests fail due to non-linearities or noise.

#### Statistical Set-up

$H_0: X \sim \mathcal{D}_d$ (full dimensional isotropy) 
((can $\mathcal{D}$ be $\mathcal{M}$?))
$H_1: X \sim \mathcal{D}_{k,\epsilon}$ (collapses to $k$-dimensional structure as $\epsilon \rightarrow 0$)

where the following are true:
* all live in $\mathbb{R}^d$,
* all have identical first moments,
* they differ only in covariance (supporting dimensions (PCA useful here?)).

### Datasets:

#### Base Manifolds

* Null A) Gaussian
* Null B) Non-linear, full dimensional

#### Degenerating Covariance - Collapsed Manifold Model

* Test A) Linear: $k$-plane embedded in $\mathbb{R}^d$
* Test B) Non-linear: $k$-torus embedded in $\mathbb{R}^d$
* Test C) Non-linear: Swiss roll embedded in $\mathbb{R}^d$

#### Test Cases

1) Running sample sizes for various $n$, looking for stability.
2) Running with varying sizes of noise added.
3) Running for high dimensionality when computationally feasible.

#### Complexes Tested for Persistence

1) VR-complex 
2) Cech complex (maybe drop due to computation)
3) DTM filtration
4) Witness complex (forgot about this one, revisit Ghrist)

For each of the above, we sweep $\epsilon \in \{0.05, 0.1, 0.2, 0.5\}$,
**which controls for the strength of the collapse.** 
This allows us to 
calibrate the null hypotheses (two point clouds) for a fixed $n,d$ to 
allow the tests to compute the power of their ability to measure collapse.
The expectation is that as $\epsilon$ decreases (stronger collapse),
the persistence-based test rejects the null more often.

#### Test Statistics Computed

##### Calibration 

There are two calibrations that need to be completed.

1. The first is to calibrate $\tau$, the appropriate scale to build
the VR / DTM filtrations to keep persistence computations
at comparable resolutions across settings.
We calibrate $\tau$ (which generates tau_map.csv) according to:
- the point clouds included in the null class
- the number of points and dimension
- the type of filtration

Later, the tau_map.csv is used as a lookup in simulations and is used to set:
- the VR max_edge_length (radius cap)
- the DTM max_f (scale cap)

2. We then can calibrate the statistical tests so that Type I error is controlled
(or FWER after correction)
For each null distribution (p-values) are computed for the PH statistics
- Total Persistence (TP)
- Mean Tail Excess (MTE)
This is accomplished using null_simulation. 

After these steps of calibration, we can run simulations of the 
three main classes of alternatives.
This is accomplished through alt_parallel.
We can then run compare_stats to complete the initial pipeline,
followed by generating power curves using power_vs_eps.



#### Preliminary Results

Across simulated datasets, persistence-based statistics frequently fail
to reject the null hypothesis for collapsed point clouds, indicating that
persistent homology contains signal that can be used to detect 
distributional collapse (deviation from full-dimension null models).

### How to run

`./run_pipeline.sh`
