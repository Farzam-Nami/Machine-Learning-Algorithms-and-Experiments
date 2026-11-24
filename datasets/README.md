# Data setup

The Gaussian Naive Bayes, baseball/hockey word-count, normal-sample and TinyMNIST files are included in the local working package. Their redistribution terms are unknown, so they remain excluded from Git by default. The flower archive is not bundled. Obtain missing data and verify redistribution terms before adding any external data to a public repository. The following paths are relative to the repository root; notebook code accesses these locations using `../datasets/` from `notebooks/`.

| Data | Required location | Notebook |
|---|---|---|
| Gaussian Naive Bayes | `datasets/gaussian_bayes/guassianBayes.mat` (original spelling) | Bayesian Classification |
| Baseball/hockey word counts | `datasets/bag_of_words/baseball_train_set.csv`, `datasets/bag_of_words/baseball_test_set.csv`, `datasets/bag_of_words/hockey_train_set.csv`, `datasets/bag_of_words/hockey_test_set.csv` | Bayesian Classification |
| Normal sample | `datasets/normal_distribution/normal.data.txt` | Density Estimation and Regression |
| TinyMNIST | `datasets/tiny_mnist/trainData.csv`, `datasets/tiny_mnist/trainLabels.csv`, `datasets/tiny_mnist/testData.csv`, `datasets/tiny_mnist/testLabels.csv` | Feature Selection and Image Classification |
| Flower images | `datasets/flower_images/flowers-20251228T193800Z-3-001.zip` | Feature Selection and Image Classification |

The supplied normal sample contains 200 observations. The supplied MAT file has 218 training samples and 109 test samples, each with five features. The MAT file must contain `xTrain`, `yTrain`, `xTest`, `yTest`. The text matrices contain numeric word counts (50 rows and 5,822 columns per split/class); `datasets/bag_of_words/vocabulary.txt` is useful for interpretation but is not read by the classifier. The normal file contains whitespace-separated numeric observations. TinyMNIST contains 5,000 training observations and 2,500 test observations, each with 196 features (14 × 14 pixels). The labels cover all ten digit classes (0–9). The four CSV files are used directly by the notebook.

The flower notebook attempts to retrieve the configured Google Drive archive if the archive is not already present. Availability and redistribution permission have not been verified. Supply the exact archive if the download is unavailable. The extracted tree expected by the code is (relative to the repository root):

```text
datasets/flower_images/
  flowers-20251228T193800Z-3-001.zip
  flowers/
    train/
      Lilly/
      Lotus/
      Orchid/
      Sunflower/
      Tulip/
    val/
      Lilly/
      Lotus/
      Orchid/
      Sunflower/
      Tulip/
```

Iris and Digits are loaded directly from scikit-learn. The Parzen samples are defined in the notebook, while the regression and synthetic clustering datasets are generated programmatically. No external files are necessary for these experiments. AlexNet and VGG16 use pretrained torchvision weights, downloaded by torchvision when not cached.
