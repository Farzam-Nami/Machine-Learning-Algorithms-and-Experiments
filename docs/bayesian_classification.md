# Bayesian classification: methods and analysis

[Notebook](../notebooks/bayesian_classification.ipynb)

## Gaussian discriminant analysis

For Gaussian class-conditional distributions, Bayes classification chooses the largest posterior score. Up to a class-independent constant, the quadratic discriminant is

$$g_i(x)=-\frac12\log|\Sigma_i|-\frac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)+\log\pi_i.$$

With shared covariance, expanding the quadratic form cancels the common terms. The remaining linear discriminant produces a hyperplane. For two classes its normal vector is $w=\Sigma^{-1}(\mu_1-\mu_2)$ and its intercept is

$$b=-\frac12(\mu_1^T\Sigma^{-1}\mu_1-\mu_2^T\Sigma^{-1}\mu_2)+\log(\pi_1/\pi_2).$$

In the Iris PCA plane, changing the priors from equal to (0.8, 0.2) changes only the intercept by $\log 4$, extending the decision region for class 1. With separate covariances, the quadratic terms no longer cancel and boundaries can curve. The figures use the entire dataset for PCA and Gaussian fitting; they illustrate decision geometry rather than out-of-sample accuracy.

### Theoretical rejection rule

With zero cost for correct decisions, unit cost for misclassification and constant rejection cost $\lambda$, the risk of selecting class $i$ is $1-P(i\mid x)$; the risk of rejection is $\lambda$. Thus rejection is appropriate whenever $\max_i P(i\mid x)<1-\lambda$ (ties can be resolved by convention). Increasing the rejection cost shrinks the rejection region. This explains the risk–coverage tradeoff, but the provided notebook does not implement the rejection-region or 80% training-split experiment. No numerical result for that extension is claimed here.

## Gaussian Naive Bayes

Estimate class priors by empirical label frequencies and class-conditional feature means and sample variances separately. The score is

$$\log p(c)+\sum_j\log \mathcal{N}(x_j;\hat\mu_{jc},\hat v_{jc}).$$

The implementation adds a small positive variance offset and compares log scores rather than multiplying densities. The recorded evaluation gives:

| Metric | Value |
|---|---:|
| Accuracy | 0.834862 |
| Precision, class 1 | 0.951220 |
| Recall, class 1 | 0.975000 |
| Precision, class 5 | 0.875000 |
| Recall, class 5 | 0.777778 |

Class 1 has both higher precision and higher recall than class 5. In particular, the lower recall for class 5 indicates a higher missed-positive fraction. No full per-class report is available in the recorded outputs.

## Multinomial Naive Bayes for text

The dataset contains 50 training and 50 test documents per class, with 5,822 word-count features. Class priors are equal. For class-specific total count $N_c$, word count $n_{wc}$ and vocabulary size $V$, the implemented estimates are

$$\hat p_{wc}^{\mathrm{MLE}}=\frac{n_{wc}}{N_c},\qquad \hat p_{wc}^{\mathrm{smooth}}=\frac{n_{wc}+\beta}{N_c+V\beta}.$$

The second expression is additive smoothing, also the posterior mean under a symmetric Dirichlet prior with concentration $\beta>0$ (or the interior MAP formula with concentration $\beta+1$). This distinction matters when interpreting the terminology in the implementation.

In the unsmoothed case, a word with a positive document count and zero estimated class probability should force its class likelihood to zero. The code instead masks out such terms; it avoids undefined arithmetic, but changes the unsmoothed model score. The reported 18% accuracy and prediction counts (64 baseball, 36 hockey) belong to that implementation, not necessarily to a canonical unsmoothed multinomial classifier.

| Pseudocount | Accuracy |
|---:|---:|
| 0 | 18% |
| 0.00000001 | 87% |
| 0.000001 | 86% |
| 0.001 | 86% |
| 0.01 | 87% |
| 0.1 | 87% |
| 1 | 79% |
| 2 | 65% |
| 5 | 55% |
| 10 | 54% |

Small positive pseudocounts remove zero probabilities; large pseudocounts pull class distributions toward uniformity and reduce contrast. Zero cannot appear on a logarithmic horizontal axis, but is included in the numerical table.

| Training documents per class | Accuracy at beta = 0.1 |
|---:|---:|
| 1 | 46% |
| 5 | 59% |
| 10 | 75% |
| 15 | 60% |
| 20 | 71% |
| 25 | 81% |
| 30 | 81% |
| 35 | 83% |
| 40 | 84% |
| 45 | 88% |
| 50 | 87% |

More observations generally improve probability estimation, but a nested prefix of documents need not produce monotonically improving classification. Both comparisons reuse the same test data, so choosing beta or training size from these results is exploratory rather than independent model validation.
