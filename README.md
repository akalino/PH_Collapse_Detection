# PH_Collapse_Detection

#### Goal
Use persistent homology to detect degenerating covariance / 
collapse to lower intrinsic dimension.

At some juncture in studying the topology of point clouds, there becomes 
less interest in studying the actual topology, and the question turns to

Can we detect, quantify, or test that collapse is happening using 
persistence?

#### Basic setup

$X_i = Y_i + \epsilon Z_i$ where $Y_i \sim \mu$ is supported on some
$k$-manifold $\mathcal{M} \subset \mathbb{R}^d$ and 
$Z_i \sim \mathcal{N}(0, \Sigma)$ with $\epsilon \rightarrow 0$

We're interested in when the covariance matrix $\Sigma_\epsilon$ 
collapses ($\rightarrow \inf$),
or **equivalently when the effective dimension drops from $d$ to $k$**.

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
* they differ only in covariance (supporting dimensions (PCA useful here)).

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

#### Test Statistics Computed


#### Preliminary Results

Across simulated datasets, persistence-based statistics frequently fail
to reject the null hypothesis for collapsed point clouds, indicating that
persistent homology contains signal that can be used to detect 
distributional collapse (deviation from full-dimension null models).

### How to run

1) Generate point clouds using `synthetic_data/point_clouds.py`. Relevant hyperparameters can be found in the YAML configs.
2) 