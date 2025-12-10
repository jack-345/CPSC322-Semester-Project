import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from mysklearn import myevaluation
from mysklearn.myclassifiers import MyNaiveBayesClassifier

# Naive Bayes unit tests
# note: order is actual/received student value, expected/solution
def test_train_test_split():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_1 = [[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]]
    y_1 = [0, 1, 2, 3, 4]
    # then put repeat values in
    X_2 = [[0, 1],
       [2, 3],
       [5, 6],
       [6, 7],
       [0, 1]]
    y_2 = [2, 3, 3, 2, 2]
    test_sizes = [0.33, 0.25, 4, 3, 2, 1]
    for X, y in zip([X_1, X_2], [y_1, y_2]):
        for test_size in test_sizes:
            X_train_solution, X_test_solution, y_train_solution, y_test_solution =\
                train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)
            X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

            assert np.array_equal(X_train, X_train_solution) # order matters with np.array_equal()
            assert np.array_equal(X_test, X_test_solution)
            assert np.array_equal(y_train, y_train_solution)
            assert np.array_equal(y_test, y_test_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    test_size = 2
    X_train0_notshuffled, X_test0_notshuffled, y_train0_notshuffled, y_test0_notshuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=False)
    X_train0_shuffled, X_test0_shuffled, y_train0_shuffled, y_test0_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=True)
    # make sure shuffle keeps X and y parallel
    for i, _ in enumerate(X_train0_shuffled):
        assert y_1[X_1.index(X_train0_shuffled[i])] == y_train0_shuffled[i]
    # same random_state but with shuffle= False vs True should produce diff folds
    assert not np.array_equal(X_train0_notshuffled, X_train0_shuffled)
    assert not np.array_equal(y_train0_notshuffled, y_train0_shuffled)
    assert not np.array_equal(X_test0_notshuffled, X_test0_shuffled)
    assert not np.array_equal(y_test0_notshuffled, y_test0_shuffled)
    X_train1_shuffled, X_test1_shuffled, y_train1_shuffled, y_test1_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    assert not np.array_equal(X_train0_shuffled, X_train1_shuffled)
    assert not np.array_equal(y_train0_shuffled, y_train1_shuffled)
    assert not np.array_equal(X_test0_shuffled, X_test1_shuffled)
    assert not np.array_equal(y_test0_shuffled, y_test1_shuffled)

# test utility function
def check_folds(n, n_splits, folds, folds_solution):
    """Utility function

    n(int): number of samples in dataset
    """
    all_test_indices = []
    all_train_indices = []
    all_train_indices_solution = []
    all_test_indices_solution = []
    for i in range(n_splits):
        # make sure all indices are accounted for in each split
        curr_fold = folds[i]
        curr_train_indexes, curr_test_indexes = curr_fold
        all_indices_in_fold = curr_train_indexes + curr_test_indexes
        assert len(all_indices_in_fold) == n
        for index in range(n):
            assert index in all_indices_in_fold
        all_test_indices.extend(curr_test_indexes)
        all_train_indices.extend(curr_train_indexes)

        curr_fold_solution = folds_solution[i]
        curr_train_indexes_solution, curr_test_indexes_solution = curr_fold_solution
        all_train_indices_solution.extend(curr_train_indexes_solution)
        all_test_indices_solution.extend(curr_test_indexes_solution)

    # make sure all indices are in a test set
    assert len(all_test_indices) == n
    for index in range(n):
        assert index in all_indices_in_fold
    # make sure fold test on appropriate number of indices
    all_test_indices.sort()
    all_test_indices_solution.sort()
    assert all_test_indices == all_test_indices_solution

    # make sure fold train on appropriate number of indices
    all_train_indices.sort()
    all_train_indices_solution.sort()
    assert all_train_indices == all_train_indices_solution

def test_kfold_split():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Notes:
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    X = [[0, 1], [2, 3], [4, 5], [6, 7]]
    y = [1, 2, 3, 4]

    n_splits = 2
    for tset in [X, y]:
        folds = myevaluation.kfold_split(tset, n_splits=n_splits)
        assert len(folds) > 0
        standard_kf = KFold(n_splits=n_splits)
        sklearn_folds = list(standard_kf.split(tset))
        folds_solution = []
        # convert all solution numpy arrays to lists
        for fold_train_indexes, fold_test_indexes in sklearn_folds:
            folds_solution.append((list(fold_train_indexes), list(fold_test_indexes)))
        check_folds(len(tset), n_splits, folds, folds_solution)

    # more complicated dataset
    table = [
        [3, 2, "no"],
        [6, 6, "yes"],
        [4, 1, "no"],
        [4, 4, "no"],
        [1, 2, "yes"],
        [2, 0, "no"],
        [0, 3, "yes"],
        [1, 6, "yes"]
    ]
    # n_splits = 2, ..., 8 (LOOCV)
    for n_splits in range(2, len(table) + 1):
        folds = myevaluation.kfold_split(table, n_splits=n_splits)
        standard_kf = KFold(n_splits=n_splits)
        # convert all solution numpy arrays to lists
        sklearn_folds = list(standard_kf.split(table))
        folds_solution = []
        # convert all solution numpy arrays to lists
        for fold_train_indexes, fold_test_indexes in sklearn_folds:
            folds_solution.append((list(fold_train_indexes), list(fold_test_indexes)))
        check_folds(len(table), n_splits, folds, folds_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    folds0_notshuffled = myevaluation.kfold_split(X, n_splits=2, random_state=0, shuffle=False)
    folds0_shuffled = myevaluation.kfold_split(X, n_splits=2, random_state=0, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(folds0_notshuffled):
        assert not np.array_equal(folds0_notshuffled[i], folds0_shuffled[i])
    folds1_shuffled = myevaluation.kfold_split(X, n_splits=2, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(folds0_shuffled):
        assert not np.array_equal(folds0_shuffled[i], folds1_shuffled[i])

# test utility function
def get_min_label_counts(y, label, n_splits):
    """Utility function
    """
    label_counts = sum([1 for yval in y if yval == label])
    min_test_label_count = label_counts // n_splits
    min_train_label_count = (n_splits - 1) * min_test_label_count
    return min_train_label_count, min_test_label_count

def test_stratified_kfold_split():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

    Notes:
        This test does not test shuffle or random_state
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    X = [[0, 1], [2, 3], [4, 5], [6, 4]]
    y = [0, 0, 1, 1]

    n_splits = 2
    folds = myevaluation.stratified_kfold_split(X, y, n_splits=n_splits)
    assert len(folds) > 0
    stratified_kf = StratifiedKFold(n_splits=n_splits)
    sklearn_folds = list(stratified_kf.split(X, y))
    folds_solution = []
    # convert all solution numpy arrays to lists
    for fold_train_indexes, fold_test_indexes in sklearn_folds:
        folds_solution.append((list(fold_train_indexes), list(fold_test_indexes)))
    # sklearn solution and order:
    # [(array([2, 3]), array([0, 1])), (array([0, 1]), array([2, 3]))]
    # fold0: TRAIN: [1 3] TEST: [0 2]
    # fold1: TRAIN: [0 2] TEST: [1 3]
    check_folds(len(y), n_splits, folds, folds_solution)
    for i in range(n_splits):
        # since the actual result could have folds in diff order, make sure this train and test set is in the solution somewhere
        # sort the train and test sets of the fold so the indices can be in any order within a set
        # make sure at least minimum count of each label in each split
        curr_fold = folds[i]
        curr_fold_train_indexes, curr_fold_test_indexes = curr_fold
        for label in [0, 1]:
            train_yes_labels = [y[j] for j in curr_fold_train_indexes if y[j] == label]
            test_yes_labels = [y[j] for j in curr_fold_test_indexes if y[j] == label]
            min_train_label_count, min_test_label_count = get_min_label_counts(y, label, n_splits)
            assert len(train_yes_labels) >= min_train_label_count
            assert len(test_yes_labels) >= min_test_label_count

    # note: this test case does not test order against sklearn's solution
    table = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    table_y = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    for n_splits in range(2, 5):
        folds = myevaluation.stratified_kfold_split(table, table_y, n_splits=n_splits)
        stratified_kf = StratifiedKFold(n_splits=n_splits)
        sklearn_folds = list(stratified_kf.split(table, table_y))
        folds_solution = []
        # convert all solution numpy arrays to lists
        for fold_train_indexes, fold_test_indexes in sklearn_folds:
            folds_solution.append((list(fold_train_indexes), list(fold_test_indexes)))
        check_folds(len(table), n_splits, folds, folds_solution)

        for i in range(n_splits):
            # make sure at least minimum count of each label in each split
            curr_fold = folds[i]
            curr_fold_train_indexes, curr_fold_test_indexes = curr_fold
            for label in ["yes", "no"]:
                train_yes_labels = [table_y[j] for j in curr_fold_train_indexes if table_y[j] == label]
                test_yes_labels = [table_y[j] for j in curr_fold_test_indexes if table_y[j] == label]
                min_train_label_count, min_test_label_count = get_min_label_counts(table_y, label, n_splits)
                assert len(train_yes_labels) >= min_train_label_count
                assert len(test_yes_labels) >= min_test_label_count

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    folds1_notshuffled = \
        myevaluation.stratified_kfold_split(X, y, n_splits=2, random_state=1, shuffle=False)
    folds1_shuffled = myevaluation.stratified_kfold_split(X, y, n_splits=2, random_state=1, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(folds1_notshuffled):
        assert not np.array_equal(folds1_notshuffled[i], folds1_shuffled[i])
    folds2_shuffled = myevaluation.stratified_kfold_split(X, y, n_splits=2, random_state=2, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(folds1_shuffled):
        assert not np.array_equal(folds1_shuffled[i], folds2_shuffled[i])

# test utility function
def check_same_lists_regardless_of_order(list1, list2):
    """Utility function
    """
    assert len(list1) == len(list2) # same length
    for item in list1:
        assert item in list2
        list2.remove(item)
    assert len(list2) == 0
    return True

def test_bootstrap_sample():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html

    Notes:
        This test does not test shuffle or random_state
    """
    np.random.seed(0)
    size = 10000
    X = [[i, i] for i in range(size)] # make a really big list of instances
    y = np.random.choice(["yes", "no", "maybe"], size=size) # made up target classes
    # doesn't matter what the instances are, bootstrap_sample should sample
    # indexes with replacement to determine which instances go in sample and out_of_bag
    X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y, random_state=1)

    # change instances to be tuples because tuples are hashable (needed for set code below)
    X_sample = [tuple(instance) for instance in X_sample]
    X_out_of_bag = [tuple(instance) for instance in X_out_of_bag]

    # check the X_sample is about ~63.2% of instances
    percent_unique = len(set(X_sample)) / size
    assert np.isclose(percent_unique, 0.632, rtol=0.1) # adjusting relative tolerance
    # to allow for larger difference than default since size is not that big (keeps code fast)
    # check the X_out_of_bag is about ~36.8% of instances
    percent_unique = len(set(X_out_of_bag)) / size
    assert np.isclose(percent_unique, 0.368, rtol=0.1)

    # check the X_sample and y_sample are parallel
    for i, instance in enumerate(X_sample):
        orig_index = X.index(list(instance)) # instance is a tuple
        assert y_sample[i] == y[orig_index]
    # check the X_out_of_bag and y_out_of_bag are parallel
    for i, instance in enumerate(X_out_of_bag):
        orig_index = X.index(list(instance)) # instance is a tuple
        assert y_out_of_bag[i] == y[orig_index]

def test_confusion_matrix():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    matrix_solution = [[2, 0, 0],
                [0, 0, 1],
                [1, 0, 2]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1, 2])
    assert np.array_equal(matrix, matrix_solution)

    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, ["ant", "bird", "cat"])
    assert np.array_equal(matrix, matrix_solution)

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]

    matrix_solution = [[0, 2],[1, 1]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1])
    assert np.array_equal(matrix, matrix_solution)

def test_accuracy_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]

    # normalize=True
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=True)
    score_sol =  accuracy_score(y_true, y_pred, normalize=True) # 0.5
    assert np.isclose(score, score_sol)

    # normalize=False
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=False)
    score_sol =  accuracy_score(y_true, y_pred, normalize=False) # 2
    assert np.isclose(score, score_sol)

def test_naive_bayes_classifier_fit():
    # test case a: in-class example
    X_train_class = [
        [1, 5],
        [2, 6],
        [1, 5],
        [1, 5],
        [1, 6],
        [2, 6],
        [1, 5],
        [1, 6]
    ]

    y_train_class = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    nb = MyNaiveBayesClassifier()
    nb.fit(X_train_class, y_train_class)

    #priors
    assert np.isclose(nb.priors["yes"], 5 / 8)
    assert np.isclose(nb.priors["no"], 3 / 8)

    #att1 posteriors
    assert np.isclose(nb.posteriors["yes"][0][1], 4 / 5)
    assert np.isclose(nb.posteriors["yes"][0][2], 1 / 5)
    assert np.isclose(nb.posteriors["no"][0][1], 2 / 3)
    assert np.isclose(nb.posteriors["no"][0][2], 1 / 3)

    #att2 posteriors
    assert np.isclose(nb.posteriors["yes"][1][5], 2/5)
    assert np.isclose(nb.posteriors["yes"][1][6], 3/5)
    assert np.isclose(nb.posteriors["no"][1][5], 2/3)
    assert np.isclose(nb.posteriors["no"][1][6], 1/3)

    # test case b: LA7 instances
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]

    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    nb_iphone = MyNaiveBayesClassifier()
    nb_iphone.fit(X_train_iphone, y_train_iphone)

    #priors
    assert np.isclose(nb_iphone.priors["yes"], 2 / 3)
    assert np.isclose(nb_iphone.priors["no"], 1 / 3)
    #posteriors
    assert np.isclose(nb_iphone.posteriors["yes"][0][1], 2 / 10)
    assert np.isclose(nb_iphone.posteriors["yes"][0][2], 8 / 10)

    # test case c: Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]

    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                     "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                     "very late", "on time", "on time", "on time", "on time", "on time"]

    nb_bramer = MyNaiveBayesClassifier()
    nb_bramer.fit(X_train_bramer, y_train_bramer)

    #priors
    assert np.isclose(nb_bramer.priors["on time"], 14 / 20)
    assert np.isclose(nb_bramer.priors["late"], 2 / 20)
    assert np.isclose(nb_bramer.priors["very late"], 3 / 20)
    assert np.isclose(nb_bramer.priors["cancelled"], 1 / 20)

    #posterior
    assert np.isclose(nb_bramer.posteriors["on time"][0]["weekday"], 9 / 14)

def test_naive_bayes_classifier_predict():
    #test case a: in-class example
    X_train = [
        [1, 5],
        [2, 6],
        [1, 5],
        [1, 5],
        [1, 6],
        [2, 6],
        [1, 5],
        [1, 6]
    ]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    nb = MyNaiveBayesClassifier()
    nb.fit(X_train, y_train)

    x_test = [[1, 5]]
    y_pred = nb.predict(x_test)

    assert y_pred[0] == "yes"

    #test case b: LA7 instances
    X_train_iphone = [
        [1, 3, "fair"], [1, 3, "excellent"], [2, 3, "fair"],
        [2, 2, "fair"], [2, 1, "fair"], [2, 1, "excellent"],
        [2, 1, "excellent"], [1, 2, "fair"], [1, 1, "fair"],
        [2, 2, "fair"], [1, 2, "excellent"], [2, 2, "excellent"],
        [2, 3, "fair"], [2, 2, "excellent"], [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes",
                      "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    nb_iphone = MyNaiveBayesClassifier()
    nb_iphone.fit(X_train_iphone, y_train_iphone)

    x_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_pred_iphone = nb_iphone.predict(x_test_iphone)

    assert y_pred_iphone[0] == "yes"
    assert y_pred_iphone[1] == "no"

    #test case c: Bramer 3.2
    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]

    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                     "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                     "very late", "on time", "on time", "on time", "on time", "on time"]
    nb_bramer = MyNaiveBayesClassifier()
    nb_bramer.fit(X_train_bramer, y_train_bramer)

    X_test_bramer = [["weekday", "winter", "high", "heavy"],]
    y_pred_bramer = nb_bramer.predict(X_test_bramer)

    assert y_pred_bramer[0] == "very late"