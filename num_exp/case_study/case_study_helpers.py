import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# helpers for decision trees
def average_depth(clf):
    """
    Returns the average depth of the decision tree classifier.

    Parameters:
    clf (DecisionTreeClassifier): A decision tree classifier trained using scikit-learn.

    Returns:
    float: The average depth of the decision tree.
    """
    def _get_depth(node, depth):
        """
        Recursively calculates the depth of each node in the tree.

        Parameters:
        node (sklearn.tree.Node): The current node being evaluated.
        depth (int): The current depth of the node.

        Returns:
        int: The maximum depth of the current node and its children.
        """
        if clf.tree_.children_left[node] == clf.tree_.children_right[node]:
            return depth
        left_depth = _get_depth(clf.tree_.children_left[node], depth + 1) if clf.tree_.children_left[node] != -1 else 0
        right_depth = _get_depth(clf.tree_.children_right[node], depth + 1) if clf.tree_.children_right[node] != -1 else 0
        return max(left_depth, right_depth)

    return _get_depth(0, 0)

def average_path_length(clf, X_test):
    """
    Returns the average path length of the decision tree classifier on a specific test set.

    Parameters:
    clf (DecisionTreeClassifier): A decision tree classifier trained using scikit-learn.
    X_test (array-like): The test set for which to compute the average path length.

    Returns:
    float: The average path length of the decision tree on the test set.
    """
    def _get_path_length(node, depth, sample):
        """
        Recursively calculates the path length of a specific sample in the tree.

        Parameters:
        node (int): The current node being evaluated.
        depth (int): The current depth of the node.
        sample (array-like): The feature values of the sample being evaluated.

        Returns:
        int: The path length of the sample in the decision tree.
        """
        if clf.tree_.children_left[node] == clf.tree_.children_right[node]:
            return depth
        feature = clf.tree_.feature[node]
        threshold = clf.tree_.threshold[node]
        if sample[feature] <= threshold:
            child_node = clf.tree_.children_left[node]
        else:
            child_node = clf.tree_.children_right[node]
        return _get_path_length(child_node, depth + 1, sample)

    path_lengths = []
    for sample in X_test:
        node = 0
        depth = 0
        path_lengths.append(_get_path_length(node, depth, sample))
    return sum(path_lengths) / len(path_lengths)

# helpers for ensemble methods
def avg_depth_ensemble(estimator):
    path_lengths = []
    for i, est in enumerate(estimator.estimators_):
        # print(est[0])
        if isinstance(est, DecisionTreeClassifier):
            path_lengths.append(average_depth(est))
        else:
            path_lengths.append(average_depth(est[0]))

    return np.mean(path_lengths)

def avg_path_length_ensemble(estimator, X_test):
    path_lengths = []
    for i, est in enumerate(estimator.estimators_):
        if isinstance(est, DecisionTreeClassifier):
            # path_lengths.append(np.mean(get_leaves(est, X_test)))
            path_lengths.append(average_path_length(est, X_test))
        else:
            path_lengths.append(average_path_length(est[0], X_test))
    return np.mean(path_lengths)