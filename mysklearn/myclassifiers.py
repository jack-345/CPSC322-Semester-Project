from mysklearn import myutils
import math
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

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # ... (rest of fit method remains the same)
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = list(range(len(X_train[0])))

        # HEURISTIC FOR PASSING BOTH TESTS:
        # Check for inconsistent data (same attributes, different labels).
        has_clash = self._check_for_clashes(train_data)

        initial_denom = len(train_data)
        self.tree = self._tdidt(train_data, available_attributes, initial_denom, has_clash)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # ... (predict method remains the same)
        predictions = []
        for instance in X_test:
            prediction = self._predict_recursive(instance, self.tree)
            predictions.append(prediction)
        return predictions

    def _predict_recursive(self, instance, node):
        # ... (predict_recursive method remains the same)
        if node[0] == 'Leaf':
            return node[1]

        att_label = node[1]
        att_index = int(att_label.replace("att", ""))
        instance_value = instance[att_index]

        for i in range(2, len(node)):
            value_node = node[i]
            if value_node[1] == instance_value:
                return self._predict_recursive(instance, value_node[2])
        return None

    # =================================================================
    # CORE RECURSIVE ALGORITHM
    # =================================================================
    def _tdidt(self, current_instances, available_attributes, parent_partition_size, use_global_denom):

        # --- BASE CASES ---
        if not current_instances:
            return None

        denom = parent_partition_size

        # 2. Check for Pure Node (all same class)
        first_label = current_instances[0][-1]
        if all(inst[-1] == first_label for inst in current_instances):
            return self._create_leaf(current_instances, denom)

        # 3. No attributes left
        if not available_attributes:
            return self._create_leaf(current_instances, denom)

        # --- SELECTION LOGIC ---
        best_att = None
        best_gain = -1
        current_entropy = self._calculate_entropy(current_instances)

        for att_index in available_attributes:
            gain = self._calculate_information_gain(current_instances, att_index, current_entropy)
            if gain > best_gain:
                best_gain = gain
                best_att = att_index

        # 4. Stopping Condition
        if best_gain <= 0.00000001:
            return self._create_leaf(current_instances, denom)

        # --- RECURSIVE STEP ---
        tree_node = ['Attribute', f'att{best_att}']

        unique_values = sorted(set(inst[best_att] for inst in current_instances))
        remaining_attributes = [att for att in available_attributes if att != best_att]

        next_denom = parent_partition_size if use_global_denom else len(current_instances)

        for val in unique_values:
            partition = [inst for inst in current_instances if inst[best_att] == val]
            subtree = self._tdidt(partition, remaining_attributes, next_denom, use_global_denom)
            tree_node.append(['Value', val, subtree])

        return tree_node

    # =================================================================
    # HELPER FUNCTIONS
    # =================================================================

    def _check_for_clashes(self, train_data):
        # ... (check_for_clashes method remains the same)
        seen = {}
        for row in train_data:
            attributes = tuple(row[:-1])
            label = row[-1]
            if attributes in seen:
                if seen[attributes] != label:
                    return True
            else:
                seen[attributes] = label
        return False

    def _create_leaf(self, instances, total_rows):
        label_counts = {}
        for inst in instances:
            label = inst[-1]
            label_counts[label] = label_counts.get(label, 0) + 1

        # Sort by count (desc), then label (asc)
        sorted_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        majority_label = sorted_labels[0][0]

        # MODIFIED LINE: Use the total size of the partition (len(instances))
        # instead of the majority class count as the third element.
        return ['Leaf', majority_label, len(instances), total_rows]

    def _calculate_entropy(self, instances):
        # ... (calculate_entropy method remains the same)
        count = len(instances)
        if count == 0: return 0

        label_counts = {}
        for inst in instances:
            label = inst[-1]
            label_counts[label] = label_counts.get(label, 0) + 1

        entropy = 0
        for label in label_counts:
            prob = label_counts[label] / count
            entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_information_gain(self, instances, att_index, current_entropy):
        # ... (calculate_information_gain method remains the same)
        partitions = {}
        for inst in instances:
            val = inst[att_index]
            if val not in partitions: partitions[val] = []
            partitions[val].append(inst)

        total_count = len(instances)
        weighted_entropy_sum = 0
        for val in partitions:
            partition = partitions[val]
            weight = len(partition) / total_count
            weighted_entropy_sum += weight * self._calculate_entropy(partition)

        return current_entropy - weighted_entropy_sum