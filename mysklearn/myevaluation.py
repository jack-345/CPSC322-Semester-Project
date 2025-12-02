import numpy as np # use numpy's random number generation
import math
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    np.random.seed(random_state)
    n = len(X)

    if isinstance(test_size, float):
        n_test = int(math.ceil(n * test_size))
    else:
        n_test = test_size

    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        test_idx = indices[-n_test:]
        train_idx = indices[:-n_test]

    X_train = [X[i] for i in train_idx]
    x_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, x_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    np.random.seed(random_state)
    n = len(X)

    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    base = n // n_splits
    extra = n % n_splits

    fold_boundaries = []
    start = 0
    for i in range(n_splits):
        size = base + (1 if i < extra else 0)
        fold_boundaries.append(indices[start:start + size].tolist())
        start += size

    folds = []
    for i in range(n_splits):
        test_idx = fold_boundaries[i]
        # Everything else is training
        train_idx = [x for j in range(n_splits) if j != i for x in fold_boundaries[j]]
        folds.append((train_idx, test_idx))

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    np.random.seed(random_state)

    class_to_indices = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(label, []).append(idx)

    if shuffle:
        for label in class_to_indices:
            np.random.shuffle(class_to_indices[label])

    strat_folds = [[] for _ in range(n_splits)]

    for label, idx_list, in class_to_indices.items():
        n = len(idx_list)
        base = n // n_splits
        extra = n % n_splits

        start = 0
        for i in range(n_splits):
            size = base + (1 if i < extra else 0)
            strat_folds[i].extend(idx_list[start:start+size])
            start += size

    folds = []
    all_indices = set(range(len(X)))

    if shuffle:
        np.random.shuffle(strat_folds)

    for i in range(n_splits):
        test_idx = strat_folds[i]
        train_idx = list(all_indices - set(test_idx))
        folds.append((train_idx, test_idx))

    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    np.random.seed(random_state)
    n_samples = len(X)

    sampled_indexes = np.random.choice(n_samples, n_samples, replace=True)
    all_indexes = set(range(n_samples))
    sample_set = set(sampled_indexes)
    out_of_bag_indexes = list(all_indexes - sample_set)

    X_sample = [X[i] for i in sampled_indexes]
    X_out_of_bag = [X[i] for i in out_of_bag_indexes]

    y_sample = [y[i] for i in sampled_indexes]
    y_out_of_bag = [y[i] for i in out_of_bag_indexes]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    n_labels = len(labels)
    matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    label_to_index = {label:i for i, label in enumerate(labels)}

    for true_label, predicted_label in zip(y_true, y_pred):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]
        matrix[true_index][predicted_index] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    if normalize:
        return correct / len(y_true) if len(y_true) > 0 else 0.0
    else:
        return correct

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the binary classification precision score

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
        pos_label(str): The label of the positive class to index the matrix

    Returns:
        precision(float): Precision of the positive class
    """
    if labels is None:
        labels = list(set(y_true))

    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0

    for true_val, pred_val in zip(y_true, y_pred):
        if pred_val == pos_label:
            if true_val == pos_label:
                tp += 1
            else:
                fp += 1

    if tp + fp == 0:
        return 0.0

    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the binary classification recall score
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
        pos_label(str): The label of the positive class to index the matrix
    """
    if labels is None:
        labels = list(set(y_true))

    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pos_label:
            if pred_val == pos_label:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0.0

    recall = tp / (tp + fn)
    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the binary classification f1 score

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
        pos_label(str): The label of the positive class to index the matrix

    Returns:
        score(float): f1 score
    """
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    if precision + recall == 0:
        return 0.0

    score = 2 * (precision * recall) / (precision + recall)
    return score