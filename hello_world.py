"""
Digit Classification with Logistic Regression and Optuna Hyperparameter Optimization

This script demonstrates the process of training a logistic regression model
for digit classification using scikit-learn and optimizing its hyperparameters
with Optuna.

It includes functions for:
- Plotting confusion matrix
- Getting hyperparameters from Optuna trials
- Performing sanity checks on the data
- Visualizing test results
- Evaluating model performance
- Showing confusion matrix
- Defining the objective function for Optuna optimization
- Main function for running the optimization process

"""


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import optuna


def plot_confusion_matrix(cm):
    """
    Plot the confusion matrix.

    Parameters:
    - cm (array): Confusion matrix to be plotted.

    Returns:
    - None
    """
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Pastel1")
    plt.title("Confusion matrix", size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(
        tick_marks,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        rotation=45,
        size=10,
    )
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel("Actual label", size=15)
    plt.xlabel("Predicted label", size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(
                str(cm[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )


def get_hyper_params_from_optuna(trial):

    """
    Get hyperparameters from Optuna trial.

    Parameters:
    - trial (optuna.Trial): Optuna trial object.

    Returns:
    - penality (str): Penalty parameter for regularization.
    - solver (str): Solver to use for optimization.
    - C (float): Inverse of regularization strength.
    - fit_intercept (bool): Whether to fit the intercept.
    - intercept_scaling (float): Scaling of the intercept.
    - l1_ratio (float or None): Elastic net mixing parameter, if penalty is 'elasticnet'.
    """

    penality = trial.suggest_categorical("penality", ["l1", "l2", "elasticnet", None])

    if penality == None:
        solver_choices = ["newton-cg", "lbfgs", "sag", "saga"]
    elif penality == "l1":
        solver = "liblinear"
    elif penality == "l2":
        solver_choices = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    elif penality == "elasticnet":
        solver = "saga"

    # Check if penalty is either 'l1' or 'elasticnet' and this code is more readable
    if penality in ("l2", None):
        solver = trial.suggest_categorical("solver_" + penality, solver_choices)

    C = trial.suggest_float("inverse_of_regularization_strength", 0.1, 1)

    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)

    if penality == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio = None
    return penality, solver, C, fit_intercept, intercept_scaling, l1_ratio


def sanity_checks(digits):
    """
    Perform sanity checks on the dataset.

    Parameters:
    - digits (sklearn.utils.Bunch): Dataset containing images and labels.

    Returns:
    - None
    """
    print("Image Data Shape", digits.data.shape)
    print("Label Data Shape", digits.target.shape)

    plt.figure(figsize=(20, 4))

    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title("Training: %i\n" % label, fontsize=20)

def visualize_test(x_test, y_test, predictions):

    """
    Visualize test results.

    Parameters:
    - x_test (array): Test feature data.
    - y_test (array): True labels of the test data.
    - predictions (array): Predicted labels of the test data.

    Returns:
    - None
    """

    for image, label, prediction in zip(x_test, y_test, predictions):
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title(f"Label: {label}, Prediction {prediction}")
        plt.show()


def evaluate_model(logistic_regression, x_test, y_test, predictions):

    """
    Evaluate the performance of the model.

    Parameters:
    - logistic_regression: Trained logistic regression model.
    - x_test (array): Test feature data.
    - y_test (array): True labels of the test data.
    - predictions (array): Predicted labels of the test data.

    Returns:
    - score (float): Mean accuracy of the model.
    - balanced_accuracy (float): Balanced accuracy of the model.
    - precesion (array): Precision scores for each class.
    - recall (array): Recall scores for each class.
    """

    score = logistic_regression.score(x_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    precesion = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    print("Mean Accuracy:", score)
    print("Balanced Accuracy:", balanced_accuracy)
    print("Precesion:", precesion)
    print("Recall:", recall)
    return score, balanced_accuracy, precesion, recall


def show_confusion_matrix(y_test, predictions):
    """
    Show the confusion matrix.

    Parameters:
    - y_test (array): True labels of the test data.
    - predictions (array): Predicted labels of the test data.

    Returns:
    - None
    """
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    plot_confusion_matrix(cm)


def objective(trial):

    """
    Objective function for Optuna optimization.

    Parameters:
    - trial (optuna.Trial): Optuna trial object.

    Returns:
    - balanced_accuracy (float): Balanced accuracy score of the model.
    """

    digits = datasets.load_digits()

    sanity_checks(digits)

    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=0
    )

    (
        penality,
        solver,
        C,
        fit_intercept,
        intercept_scaling,
        l1_ratio,
    ) = get_hyper_params_from_optuna(trial)

    logisticRegr = LogisticRegression(
        penalty=penality,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        solver=solver,
        l1_ratio=l1_ratio,
    )
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)

    visualize_test(x_test, y_test, predictions)

    _, balanced_accuracy, _, _ = evaluate_model(
        logisticRegr, x_test, y_test, predictions
    )

    show_confusion_matrix(y_test, predictions)

    return balanced_accuracy


def main():

    """
    Main function to run the optimization process.

    Parameters:
    - None

    Returns:
    - None
    """

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    trial = study.best_trial

    print("Balanced Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


main()
