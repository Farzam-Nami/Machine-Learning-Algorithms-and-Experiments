# Machine Learning Algorithms and Experiments

A collection of Python implementations and experimental analyses covering statistical machine learning, probabilistic classification, nonparametric estimation, regression, feature selection, neural networks, transfer learning and Gaussian mixture models. The notebooks combine mathematical explanations, executable implementations, visualizations and recorded experimental results.

## Notebooks

| Notebook | Main topics | Analysis |
|---|---|---|
| [Bayesian Classification](notebooks/bayesian_classification.ipynb) | LDA, QDA, Gaussian Naive Bayes, multinomial Naive Bayes, additive smoothing | [Documentation](docs/bayesian_classification.md) |
| [Density Estimation and Regression](notebooks/density_estimation_and_regression.ipynb) | Parzen windows, normal MLE, inference, Gaussian basis functions, ridge regression | [Documentation](docs/density_estimation_and_regression.md) |
| [Feature Selection and Image Classification](notebooks/feature_selection_and_image_classification.ipynb) | Forward selection, backward elimination, dense neural networks, AlexNet, VGG16 | [Documentation](docs/feature_selection_and_image_classification.md) |
| [Gaussian Mixture Models](notebooks/gaussian_mixture_models.ipynb) | K-Means, EM, initialization, singularity, covariance types, generative modeling, Bayesian mixtures | [Documentation](docs/gaussian_mixture_models.md) |

## Repository structure

```text
notebooks/     Executable notebooks with recorded outputs
datasets/      Dataset directories and setup instructions
docs/          Mathematical explanations and experimental analyses
```

## Selected results

| Experiment | Observed result |
|---|---|
| Gaussian Naive Bayes | Test accuracy: 83.49% |
| Bag-of-words classification | 18% without smoothing; up to 87% with smoothing using all training documents |
| Parzen classification | All three query points assigned to class 2 at both tested bandwidths |
| Gaussian basis regression | Minimum mean approximation error 0.011646 at ridge parameter 0.246597 |
| TinyMNIST feature selection | Forward: 79.96% with 61 features; backward: 74.80% with 97 features |
| Flower classification | Best validation accuracy: 93.37% (frozen AlexNet) |
| GMM initialization | All 50 runs converged, with different final likelihoods |
| Bayesian Gaussian mixture | Three of ten fitted weights exceed 0.01 |

These results describe the specific experimental protocols in the notebooks. In particular, feature selection reuses its test split for candidate selection, text classification reuses its test split for hyperparameter comparisons, and the image models are compared on validation rather than independent test data. Consult the analysis files for numerical and methodological caveats.

## Setup

Use a Python environment with Jupyter and the libraries listed in `requirements.txt`:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Open notebooks in `notebooks/`. Jupyter normally uses the notebook directory as the working directory. If running in an IDE, set the working directory to `notebooks/` before executing cells so that `../datasets/` resolves correctly. Generated figures and evaluation files are written relative to the notebook working directory. The notebooks contain recorded outputs and can be inspected without executing them. The local working package includes data for Gaussian Naive Bayes, bag-of-words classification, normal-distribution estimation and TinyMNIST feature selection. The flower experiment attempts to download its configured archive. See the [data setup guide](datasets/README.md) for paths and redistribution considerations. A CUDA-capable GPU is recommended for the flower experiments; pretrained weights may require an internet connection. The notebooks do not require identical hardware to be inspected, but exact numerical reproduction can depend on library versions, hardware and random operations.

## Implementation scope

Gaussian discriminants, the Naive Bayes classifiers, Parzen density estimation and Gaussian basis ridge regression use explicit numerical implementations. Scikit-learn is used for PCA, feature-selection classifier fitting and mixture models. The neural networks use PyTorch with explicit optimization and validation loops. These experiments are not a production application or a standardized benchmark.
