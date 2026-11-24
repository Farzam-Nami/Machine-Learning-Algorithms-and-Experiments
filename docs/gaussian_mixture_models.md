# Gaussian mixture models: experimental analysis

[Notebook](../notebooks/gaussian_mixture_models.ipynb)

## Clustering geometry

K-Means minimizes $\sum_n\|x_n-\mu_{c(n)}\|^2$. Its assignments are determined by nearest Euclidean center, leading to Voronoi decision regions. By contrast, the Gaussian mixture models a normalized density

$$p(x)=\sum_{k=1}^K\pi_k\mathcal{N}(x;\mu_k,\Sigma_k),\qquad \pi_k\ge0,\quad \sum_k\pi_k=1.$$

EM alternates responsibility estimation and weighted parameter updates. Full covariance captures component-specific orientation and anisotropy. In the transformed 1,500-point three-cluster example, GMM's fitted weights are approximately 0.333227, 0.333426 and 0.333347, with average likelihood lower bound -2.460677. The comparison demonstrates the role of covariance geometry; no external accuracy metric was calculated.

## Optimization landscape

For the overlapping 500-point four-center dataset, 50 random single-start mixtures are fitted. EM can improve the likelihood until a stopping condition is reached, but the mixture likelihood is nonconvex. Multiple attained likelihood levels can represent different basins or stopping points rather than a global optimum.

| Statistic | Value |
|---|---:|
| Converged fits | 50/50 |
| Minimum lower bound | -6.498144 |
| Median | -6.497578 |
| Mean | -6.447312 |
| Maximum | -5.864748 |
| Standard deviation | 0.170318 |
| Minimum / median / maximum iterations | 2 / 2 / 42 |
| Mean iterations | 4.94 |

The spread demonstrates initialization sensitivity even when all fits report convergence. Practical mitigation includes multiple starts, selecting the best attained likelihood and considering informed initializations. These measures improve the search but do not prove global optimality.

## Singularity and regularization

A Gaussian density includes the factor $|\Sigma|^{-1/2}$. At $x=\mu$, the exponent equals zero, so collapsing covariance can cause unbounded density. In a mixture, one component can collapse on a single observation while others model the remaining observations. This creates a degeneracy of unconstrained maximum likelihood.

Appending the outlier [10, 10] to 100 synthetic observations and forcing the specified initial means causes the unregularized model to raise a covariance-related `ValueError`. Adding diagonal regularization of $10^{-6}$ allows fitting to finish, with lower bound -4.813895 and a fitted mean at the outlier. The covariance offset prevents the zero-variance collapse, but does not by itself establish that the outlier deserves its own component.

## Covariance constraints

| Structure | Geometry | Covariance parameters for K components in d dimensions |
|---|---|---|
| Spherical | Component-specific isotropic variance | K |
| Diagonal | Component-specific axis-aligned ellipsoid | Kd |
| Tied | Shared full covariance | d(d+1)/2 |
| Full | Component-specific unrestricted covariance | Kd(d+1)/2 |

Full covariance can model different rotated ellipses, but parameter estimation becomes more expensive and less stable when data are limited. A simpler covariance structure can trade some approximation flexibility for lower variance. The plots compare the resulting geometry; no independent likelihood criterion or clustering metric is reported for ranking covariance structures.

## Generative modeling

The Digits dataset contains 1,797 flattened 8 × 8 images. A 30-component full-covariance GMM is fitted without labels. Its reported average training likelihood lower bound is 36.711061. Ten vectors of length 64 are sampled and reshaped into images. Unlike a discriminative classifier that estimates a decision rule or conditional class probability, the GMM models $p(x)$ and supports direct sampling. K-Means alone lacks a learned within-cluster distribution. The generated samples are qualitative; pixel-range constraints and an independent generative evaluation are not implemented.

## Bayesian mixtures

A symmetric finite Dirichlet prior with concentration below one favors sparse mixing proportions near the simplex boundary; concentration one gives a uniform density over the simplex, and concentration above one favors balanced proportions. The implemented `BayesianGaussianMixture` uses scikit-learn's default truncated Dirichlet-process prior, with small concentration 0.001 and an upper limit of ten components. It is therefore not exactly the symmetric finite Dirichlet model, although low concentration also encourages sparse effective weights.

The learned weights are approximately:

| Component index | Weight |
|---:|---:|
| 0 | 0.00201389 |
| 1 | 0.00200983 |
| 2 | 0.33395004 |
| 3 | 0.00199972 |
| 4 | 0.32999315 |
| 5 | 0.00198170 |
| 6 | 0.00196980 |
| 7 | 0.00195797 |
| 8 | 0.32412197 |
| 9 | 0.00000193 |

Three components exceed the descriptive threshold of 0.01, matching the number of centers in the synthetic dataset. This is an empirical observation, not a guarantee of recovering the true component count in arbitrary settings.
