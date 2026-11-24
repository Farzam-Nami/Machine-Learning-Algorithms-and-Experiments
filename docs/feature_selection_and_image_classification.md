# Feature selection and image classification

[Notebook](../notebooks/feature_selection_and_image_classification.ipynb)

## Feature conditioning

TinyMNIST has 5,000 training and 2,500 test samples and 196 input features. Correct Classification Rate (CCR) is the percentage of correct predictions. At each forward step, all candidate one-feature additions are evaluated using GaussianNB, and the highest-scoring addition is retained. Backward elimination evaluates one-feature removals from the current subset and retains the best removal.

| Method | Best feature count | Maximum CCR |
|---|---:|---:|
| Forward selection | 61 | 79.96% |
| Backward elimination | 97 | 74.80% |

The forward search attains a 5.16 percentage-point higher observed score with 36 fewer features (about 31% of all input features). The difference illustrates how greedy search direction changes the subsets visited. Forward selection does not prove a globally optimal subset, and backward elimination does not include the all-feature baseline in its recorded trajectory. Selected feature indices are stored in the generated DataFrames; the saved displayed best rows are truncated, so a complete subset listing cannot be recovered from those rows alone.

Both searches repeatedly use the designated test set to select candidate subsets and report their score. Consequently, the maxima are selection-set performance and may be optimistically biased. A separate validation set or nested cross-validation with an untouched test split would be needed to estimate generalization. Some candidate evaluations also emit divide-by-zero and invalid-value warnings because of degenerate feature variances; these warnings limit interpretation of individual candidate scores.

## Flower classification protocol

Five classes (Lilly, Lotus, Orchid, Sunflower and Tulip) are supplied in separate training and validation folders. Images are resized to 224 × 224, converted to tensors and normalized with ImageNet statistics. Random horizontal flips are applied only to training images. Models use Adam, batch size 64, learning rate 0.001 and cross-entropy on logits, with explicit training and evaluation loops.

| Model | Architecture and optimization |
|---|---|
| Linear | Flattened RGB pixels mapped directly to five logits; 10 epochs |
| ReLU MLP | 128 hidden units and ReLU; 10 epochs |
| Improved MLP | 256 hidden units, BatchNorm, ReLU, dropout 0.5, weight decay 0.0001, patience 5; 30 epochs |
| AlexNet | Pretrained weights frozen; replace and train final classification layer; 10 epochs |
| VGG16 | Same frozen-final-layer protocol; 10 epochs |

The final classification layers produce logits, as PyTorch CrossEntropyLoss includes log-softmax. Transfer learning adapts the final layer without updating frozen pretrained parameters. The code calls `model.train()` during training, so existing classifier dropout remains active despite frozen weights. This is distinct from fine-tuning the backbone.

## Recorded comparison

| Model | Best validation accuracy (%) | Final validation accuracy (%) | Minimum validation loss | Epochs |
|---|---:|---:|---:|---:|
| Linear | 52.522255 | 50.247280 | 11.775347 | 10 |
| ReLU (128) | 54.896142 | 50.445104 | 3.928170 | 10 |
| Improved MLP | 61.028684 | 59.545005 | 1.017784 | 30 |
| AlexNet frozen | 93.372898 | 91.691395 | 0.231571 | 10 |
| VGG16 frozen | 88.229476 | 88.229476 | 0.389444 | 10 |

The dense nonlinear baseline improves the best validation accuracy modestly relative to the linear model. Its validation accuracy fluctuates and ends below its peak. The improved MLP reaches 61.03%, with a much lower recorded cross-entropy than either simpler dense model. However, hidden width and training duration also differ; the comparison cannot isolate the effect of dropout, weight decay or batch normalization individually. Early stopping is enabled, but the model completes all 30 epochs and restores the best validation-loss state. The reported final accuracy is the last history entry, not a separate evaluation after restoration.

The two pretrained convolutional models are markedly stronger on the recorded validation split. AlexNet reaches 93.37% at epoch 8, versus VGG16's 88.23% at epoch 10. AlexNet ends with 91.69% validation accuracy; VGG16's highest and final recorded accuracy coincide. These differences are consistent with the utility of pretrained visual features, but the ordering is conditional on this dataset, frozen-head configuration and optimization budget. Peak accuracy and minimum loss need not occur at the same epoch. There is no independent test-set estimate or repeated-seed confidence interval.
