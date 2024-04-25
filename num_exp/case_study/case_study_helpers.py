import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")


def average_depth(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # Convert lists to numpy arrays for easy manipulation
    is_leaves = np.array(is_leaves)
    node_depth = np.array(node_depth)

    # Filter node_depth based on the indices where is_leaves is True
    node_depth_leaves = node_depth[is_leaves]

    # return the mean of node_depth_leaves
    return np.mean(node_depth_leaves)

'''
# helpers for decision trees
def average_depth1(clf):
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
'''

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