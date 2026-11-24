# Density estimation, statistical inference and regression

[Notebook](../notebooks/density_estimation_and_regression.ipynb)

## Gaussian Parzen windows

For dimension $d=3$, ten samples in each class and spherical bandwidth $h$, the estimated class density is

$$\hat p(x\mid\omega_k)=\frac{1}{N_k(2\pi)^{d/2}h^d}\sum_{i=1}^{N_k}\exp\left(-\frac{\|x-x_i^{(k)}\|^2}{2h^2}\right).$$

Equal priors reduce the classification rule to the maximum estimated class density.

| Bandwidth | Query | Density class 1 | Density class 2 | Density class 3 | Prediction |
|---:|---|---:|---:|---:|---|
| 1 | (0.50, 1.00, 0.00) | 0.00799505 | 0.02991049 | 0.02396666 | 2 |
| 1 | (0.31, 1.51, -0.50) | 0.00974237 | 0.03065449 | 0.01301876 | 2 |
| 1 | (-0.30, 0.44, -0.10) | 0.00888328 | 0.02401674 | 0.01124259 | 2 |
| 0.1 | (0.50, 1.00, 0.00) | 5.5714e-21 | 4.2979e-6 | 5.0796e-18 | 2 |
| 0.1 | (0.31, 1.51, -0.50) | 1.8230e-21 | 7.6204e-7 | 1.3710e-26 | 2 |
| 0.1 | (-0.30, 0.44, -0.10) | 2.3093e-13 | 2.4020e-5 | 6.7559e-41 | 2 |

The predictions are unchanged in this example, but density magnitudes differ greatly. Small bandwidth emphasizes local neighborhoods; larger bandwidth smooths the estimated density. With very small bandwidths, underflow may become relevant. These query points do not constitute a labeled accuracy benchmark.

## Maximum likelihood for a normal sample

For $n=200$ independent observations with mean $\mu$ and variance $v$, minimizing the negative log-likelihood

$$L(\mu,v)=\frac n2\log(2\pi v)+\frac{1}{2v}\sum_i(x_i-\mu)^2$$

leads to $\hat\mu=\bar x$ and $\hat v=n^{-1}\sum_i(x_i-\bar x)^2$. The score equations give these stationary points; for fixed variance the mean minimizes squared deviations, and substituting the mean yields the unique positive variance optimum for nonconstant data.

| Quantity | Recorded value |
|---|---:|
| Mean MLE | 100.480500 |
| Variance MLE | 216.318470 |
| Variance grid minimum | 216.588868 |
| NLL at MLE | 821.462879 |
| NLL at shifted pair | 822.323243 |
| Unbiased sample variance | 217.405497 |

The grid discrepancy follows from discretization. The inverse Fisher-information estimate is diagonal, with entries $\hat v/n$ and $2\hat v^2/n$: 1.08159235 and 467.93680355. The alternative plug-in using unbiased sample variance gives 1.08702749 and 472.65150229. Neither comparison alone establishes superior estimation accuracy.

### Inference

The implemented variance confidence interval uses chi-square degrees of freedom $n$ and gives (179.474287, 265.865116). When the mean is estimated from the same normal sample, the exact pivot is $n\hat v/v\sim\chi^2_{n-1}$. The recorded result is retained as an approximation, not the exact interval.

For testing $H_0:\mu=103$, the Z statistic is $(\hat\mu-103)/\sqrt{\hat v/n}$. The likelihood-ratio statistic is $n\log(\hat v_0/\hat v)$ where $\hat v_0=n^{-1}\sum_i(x_i-103)^2$. The Wald statistic is the square of Z in this setting.

| Test | Statistic | p-value |
|---|---:|---:|
| Z | -2.422605 | 0.015410 |
| LR | 5.784549 | 0.016168 |
| Wald | 5.869014 | 0.015410 |

The three asymptotic tests agree on rejection at 5%, not at 1%. For positive scaling $a$, $\mathrm{CV}(aX)=a\sigma/(a\mu)=\mathrm{CV}(X)$. The invariance principle gives $\widehat{\mathrm{CV}}=\sqrt{\hat v}/\hat\mu=0.146374$. The gradient of $g(\mu,v)=\sqrt v/\mu$ is $(-\sqrt v/\mu^2,1/(2\mu\sqrt v))^T$. Using the delta method, the standard error is 0.007474 and the 95% interval is (0.131726, 0.161023).

## Gaussian basis regression

The simulated signal is normalized sinc, with 25 uniformly sampled inputs and additive Gaussian noise of variance 0.01. A bias column and 21 Gaussian basis functions of standard deviation 0.04 yield a design matrix with 22 columns. With $R=\mathrm{diag}(0,1,\dots,1)$, ridge minimizes

$$\|y-H\beta\|^2+\lambda\beta^TR\beta.$$

The solution satisfies $(H^TH+\lambda R)\beta=H^Ty$. Weak regularization can fit noise; strong regularization shrinks basis coefficients and can oversmooth the signal. The three displayed fits compare lambda 0.001, 0.05 and 5, including a mean fit from 100 simulated datasets. The grid experiment evaluates 100 independently generated datasets per lambda against the noiseless function at 100 points. Its observed optimum is lambda 0.2465969639, with minimum average squared approximation error 0.0116459010. The intermediate optimum reflects the bias–variance tradeoff under this specific simulation. It is not a universal ridge setting.
