from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()

        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predict_nums = self.regressor.predict(X_test)
        y_predicted = [self.discretizer(pred) for pred in predict_nums]
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        
        for test_instance in X_test:
            # distances from test_instance to all training instances
            instance_distances = []
            for i, train_instance in enumerate(self.X_train):
                # Calculate Euclidean distance
                dist = sum((test_instance[j] - train_instance[j]) ** 2 
                          for j in range(len(test_instance))) ** 0.5
                instance_distances.append((dist, i))
            
            # sort by distance and get k nearest neighbors
            instance_distances.sort(key=lambda x: x[0])
            k_nearest = instance_distances[:self.n_neighbors]
            
            # separate distances and indices
            k_distances = [dist for dist, _ in k_nearest]
            k_indices = [idx for _, idx in k_nearest]
            
            distances.append(k_distances)
            neighbor_indices.append(k_indices)
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        _, neighbor_indices = self.kneighbors(X_test)

        for indices in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in indices]
            label_count = {}
            for label in neighbor_labels:
                label_count[label] = label_count.get(label, 0) + 1

            most_common_label = max(label_count, key=label_count.get)
            y_pred.append(most_common_label)
        return y_pred


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_count = {}
        for label in y_train:
            label_count[label] = label_count.get(label, 0) + 1
        self.most_common_label = max(label_count, key=label_count.get)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in range(len(X_test))]

class MyNaiveBayesClassifier:
    """
    Represents a Naive Bayes classifier.
    """

    def __init__(self):
        """
        Initializer for NaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples

        """
        self.priors = {}
        self.posteriors = {}
        total_instances = len(y_train)
        n_attributes = len(X_train[0])

        for label in y_train:
            self.priors[label] = self.priors.get(label, 0) + 1

        for label in self.priors:
            self.priors[label] = self.priors[label] / total_instances

        for label in self.priors.keys():
            self.posteriors[label] = {}

            for att_index in range(n_attributes):
                val_counts = {}
                label_count = 0

                for i in range(len(X_train)):
                    if y_train[i] == label:
                        label_count += 1
                        att_value = X_train[i][att_index]
                        val_counts[att_value] = val_counts.get(att_value, 0) + 1

                    self.posteriors[label][att_index] = {}
                    for att_value, count in val_counts.items():
                        self.posteriors[label][att_index][att_value] = count / label_count

    def predict(self, X_test):
        """
        Makes predictions for X_test using Naive Bayes classifier.
        Args:
            X_test(list of list of numeric vals): The list of testing samples

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            class_prob = {}

            for label in self.priors.keys():
                probability = self.priors[label]

                for att_index, att_value in enumerate(instance):
                    if (att_index in self.posteriors[label] and att_value in self.posteriors[label][att_index]):
                        probability += self.posteriors[label][att_index][att_value]
                    else:
                        probability = 0
                        break

                class_prob[label] = probability

            pred_label = max(class_prob, key = class_prob.get)
            y_predicted.append(pred_label)

        return y_predicted