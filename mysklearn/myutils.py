import numpy as np # use numpy's random number generation
from mysklearn import myevaluation


# TODO: your reusable general-purpose functions here
def random_subsample(X, y, classifier, k=10, test_size=0.33, random_state_base=0):
    """Perform random subsampling evaluation

    Args:
        X(list of list): The feature data
        y(list): The target values
        classifier: A classifier object with fit() and predict() methods
        k(int): Number of iterations
        test_size(float or int): Size of test set
        random_state_base(int): Base random state for reproducibility

    Returns:
        tuple: (mean_accuracy, mean_error_rate)
    """
    accuracies = []

    for i in range(k):
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(
            X, y, test_size=test_size, random_state=random_state_base + i, shuffle=True
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = myevaluation.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, 1 - mean_accuracy


def cross_val_predict(X, y, classifier, n_splits=10, stratify=False, random_state=0):
    """Perform k-fold cross validation

    Args:
        X(list of list): The feature data
        y(list): The target values
        classifier: A classifier object with fit() and predict() methods
        n_splits(int): Number of folds
        stratify(bool): If True, use stratified k-fold
        random_state(int): Random state for reproducibility

    Returns:
        tuple: (accuracy, error_rate, all_y_true, all_y_pred)
    """
    if stratify:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits, random_state=random_state, shuffle=True
        )
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits, random_state=random_state, shuffle=True
        )

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    accuracy = myevaluation.accuracy_score(all_y_true, all_y_pred)
    return accuracy, 1 - accuracy, all_y_true, all_y_pred

def bootstrap_method(X, y, classifier, k=10, random_state_base=0):
    """Perform bootstrap method evaluation

    Args:
        X(list of list): The feature data
        y(list): The target values
        classifier: A classifier object with fit() and predict() methods
        k(int): Number of bootstrap iterations
        random_state_base(int): Base random state for reproducibility

    Returns:
        tuple: (mean_accuracy, mean_error_rate)
    """
    accuracies = []

    for i in range(k):
        X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(
            X, y, random_state=random_state_base + i
        )

        # Only evaluate on out-of-bag samples
        if len(X_test) > 0:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = myevaluation.accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, 1 - mean_accuracy


def discretize_mpg(mpg):
    """Convert mpg to DOE rating

    Args:
        mpg(float): Miles per gallon value

    Returns:
        int: DOE rating (1-10)
    """
    if mpg >= 45:
        return 10
    elif mpg >= 37:
        return 9
    elif mpg >= 31:
        return 8
    elif mpg >= 27:
        return 7
    elif mpg >= 24:
        return 6
    elif mpg >= 20:
        return 5
    elif mpg >= 17:
        return 4
    elif mpg >= 15:
        return 3
    elif mpg >= 14:
        return 2
    else:
        return 1


def format_confusion_matrix(y_true, y_pred, labels, label_name="Labels"):
    """Format a confusion matrix for display with tabulate

    Args:
        y_true(list): True labels
        y_pred(list): Predicted labels
        labels(list): List of all possible labels
        label_name(str): Label name

    Returns:
        tuple: (table_data, headers) ready for tabulate
    """
    matrix = myevaluation.confusion_matrix(y_true, y_pred, labels)

    # Prepare data for tabulate
    table_data = []
    for i, label in enumerate(labels):
        row = [label] + matrix[i] + [sum(matrix[i])]
        table_data.append(row)

    # Add totals row
    totals_row = ["Total"] + [sum(matrix[i][j] for i in range(len(labels)))
                              for j in range(len(labels))]
    totals_row.append(sum(totals_row[1:]))
    table_data.append(totals_row)

    # Create headers
    headers = [label_name] + [str(label) for label in labels] + ["Total"]

    return table_data, headers