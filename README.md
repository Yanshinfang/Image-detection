# Image-detection
# Task A

## Part 1: Load Image

Ensure the ability to load images.

## Part 2: Understanding Models

### Parametric vs Non-parametric Models

- **Parametric Models**: Linear models that determine parameters based on the data. They are easier to fit due to considering linear relationships but may struggle with predicting unknown data.
- **Non-parametric Models**: Find various approximations closely matching the actual data. They can predict real-world scenarios with better accuracy but are harder to interpret.

### Ensemble Learning

Ensemble learning integrates multiple supervised learning models to create a stronger, more accurate model.

- **Bagging**: Samples training data to create different subsets, resulting in independently parallelized models. Example: Random Forest.
- **Boosting**: Assigns different weights to each training data based on difficulty. Each subsequent model focuses on improving where the previous one performed poorly. Example: AdaBoost.
- **Stacking**: Combines multiple unrelated models (potentially different algorithms) by using their predictions as features to train another model for final prediction.

### Parameters Explanation

- **n_neighbors in KNeighborsClassifier**: Represents the number of nearest neighbors considered for classification. The number of neighbors affects classification prediction.
- **n_estimators in RandomForestClassifier and AdaBoostClassifier**: Represents the number of base learners or trees in the ensemble.

### Confusion Matrix

The confusion matrix consists of four numbers:

- **True Positive (TP)**: Predicted positive and actual positive.
- **False Positive (FP)**: Predicted positive but actual negative.
- **False Negative (FN)**: Predicted negative but actual positive.
- **True Negative (TN)**: Predicted negative and actual negative.

### Precision and Recall

Precision and Recall are used when dealing with imbalanced target variables. They are calculated as follows:

- **Precision**: tp/(tp+fp), measures the accuracy of positive predictions.
- **Recall**: tp/(tp+fn), measures the ability to find all positive samples.

## Part 3: Model Parameters

### KNN

- n_neighbors: 3 (best)

### Random Forest

- n_estimators: 100 (best)

### AdaBoost

- n_estimators: 150 (best)

# Task B

## Part 1: Show Result

Display the result.

## Part 2: Experiment Results

After 25 epochs with a batch size of 10:

- Final mAP: 0.911
- Precision: 0.851
- Recall: 0.88

Using the best.pt file resulted in issues (FN consistently equals 1), potentially due to uncleaned parameters, so the last.pt file was selected.

Train Accuracy: 0.908 (>90)
Test Accuracy: 0.928 (>90)

## Problems Encountered

- Using best.pt resulted in issues (FN consistently equals 1), potentially due to uncleaned parameters, so the last.pt file was selected.
- Insufficient GPU resources in Colab, necessitating switching accounts to fine-tune parameters.

