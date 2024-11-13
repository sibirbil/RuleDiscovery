import sys
# sys.path.insert(1,'./col_gen_estimator/col_gen_estimator')
# sys.path.insert(1,'./col_gen_estimator/col_gen_estimator')
# sys.path.insert(1,'./col_gen_estimator/examples')
import numpy as np
import random
from time import time
from sklearn.model_selection import train_test_split
from sklearn import tree

from _col_gen_classifier import *
from _bdr_classifier import *
from _dtree_classifier import *
from _parameter_tuner import *
from dtree_experiment import get_is_leaves, get_all_splits, merge_all_splits, get_all_nodes, merge_all_nodes, get_splits_list, \
    get_all_leaves_paths, get_paths_list, get_leaves_list, get_nodes_list, add_rows_to_splits, get_path_for_leaf, preprocess, reduce_splits


# get average rule length of DT heuristic decision tree
def DT_heuristic_avg_rule_length(tree):
    path_lengths = []
    for path in tree.master_problem.selected_paths:
        path_lengths.append(len(path.splits))
    return np.mean(path_lengths)


def DT_heuristic_avg_rule_length_sample(tree, X):
    """Predicts the class based on the solution of master problem.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples. The inputs should only contain numeric values.

    Returns
    -------
    y : ndarray, shape (n_samples,)
        The label for each sample.
    """
    from sklearn.utils.validation import check_array, check_is_fitted
    from _dtree_classifier import row_satisfies_path
    X = check_array(X, accept_sparse=True)
    check_is_fitted(tree, 'is_fitted_')
    selected_paths = tree.master_problem.selected_paths
    # Check for each row, which path it satisfies. There should be exactly
    # one.
    num_rows = X.shape[0]
    y_predict = np.zeros(X.shape[0], dtype=int)
    path_length_samples = []
    for row in range(num_rows):
        for path in selected_paths:
            if row_satisfies_path(X[row], tree.leaves[path.leaf_id],
                                  tree.splits, path):
                y_predict[row] = path.target
                path_length_samples.append(len(path.splits))
                break
    return np.mean(path_length_samples)

def run_DT_heuristic(X_train, y_train, experiment_max_depth, time_limit = 300, preprocessing_time_limit = 100,
                     random_state = 21, REDUCED_SPLITS = True, max_iterations = -1):
    # experiment_max_depth = pgrid['max_depth']
    use_old_sp = False
    # preprocessing_time_limit = 100
    random.seed(random_state)
    # train_file = ""
    # test_file = ""

    is_large_dataset = False
    if X_train.shape[0] >= 5000:
        is_large_dataset = True

    t0 = time()

    combined_splits = {}
    combined_nodes = {}
    max_num_nodes = 2 ** experiment_max_depth - 1
    for node_id in range(max_num_nodes):
        node = Node()
        node.id = node_id
        node.candidate_splits = []
        combined_nodes[node_id] = node
    q_root = int(150 / max_num_nodes)
    q_node = int(100 / max_num_nodes)

    for i in range(300):
        if preprocessing_time_limit > 0 and time() - t0 > preprocessing_time_limit:
            break
        X_train_train, _, y_train_train, _ = train_test_split(
            X_train, y_train, test_size=0.1, random_state=i)
        dt = tree.DecisionTreeClassifier(
            max_depth=experiment_max_depth, random_state=99)
        clf_dt = dt.fit(X_train_train, y_train_train)

        n_nodes = clf_dt.tree_.node_count
        children_left = clf_dt.tree_.children_left
        children_right = clf_dt.tree_.children_right
        feature = clf_dt.tree_.feature
        threshold = clf_dt.tree_.threshold
        targets = clf_dt.tree_.value
        is_leaves = get_is_leaves(n_nodes, children_left, children_right)
        all_splits = get_all_splits(
            n_nodes, feature, threshold, is_leaves)
        combined_splits = merge_all_splits(combined_splits, all_splits)
        all_nodes = get_all_nodes(
            children_left, children_right, n_nodes,
            feature, threshold, is_leaves, combined_splits)
        combined_nodes = merge_all_nodes(combined_nodes, all_nodes)

    print("DecisionTree")
    d_tree_start_time = time()
    dt = tree.DecisionTreeClassifier(
        max_depth=experiment_max_depth, random_state=99)

    clf_dt = dt.fit(X_train, y_train)
    train_accuracy = clf_dt.score(X_train, y_train)
    # test_accuracy = clf_dt.score(X_test, y_test)

    # print("Train Acurracy: ", train_accuracy)
    # print("Test Acurracy: ", test_accuracy)
    t1 = time()
    # print("time elapsed: ", t1 - d_tree_start_time)

    # cart_results = Results()
    # cart_results.name = "CART"
    # cart_results.train_accuracy = train_accuracy
    # cart_results.test_accuracy = test_accuracy
    # cart_results.training_time = t1 - d_tree_start_time

    n_nodes = clf_dt.tree_.node_count
    children_left = clf_dt.tree_.children_left
    children_right = clf_dt.tree_.children_right
    feature = clf_dt.tree_.feature
    threshold = clf_dt.tree_.threshold

    is_leaves = get_is_leaves(n_nodes, children_left, children_right)

    # Create all used splits
    all_splits = get_all_splits(n_nodes, feature, threshold, is_leaves)

    combined_splits = merge_all_splits(combined_splits, all_splits)
    splits = get_splits_list(combined_splits)
    # Create node and add correspondning split to candidate splits
    all_nodes = get_all_nodes(
        children_left, children_right, n_nodes, feature, threshold, is_leaves,
        combined_splits)
    combined_nodes = merge_all_nodes(combined_nodes, all_nodes, 1000)

    # If some node never appeared, populate its candidate splits here.
    for node_id in range(max_num_nodes):
        if not combined_nodes[node_id].candidate_splits:
            combined_nodes[node_id].candidate_splits.append(0)
            combined_nodes[node_id].candidate_splits_count.append(0)
            combined_nodes[node_id].last_split = 0

    all_leaves, all_paths = get_all_leaves_paths(
        combined_nodes, experiment_max_depth, splits, X_train,
        y_train)

    paths = get_paths_list(all_paths)
    leaves = get_leaves_list(all_leaves)
    splits = get_splits_list(combined_splits)
    nodes = get_nodes_list(combined_nodes)

    if REDUCED_SPLITS:
        nodes = reduce_splits(nodes, splits, q_root, q_node)

    # for split in splits:
    #     print("Split removed ", split.id, split.removed)

    # Add more paths for initialization.
    for i in range(100):
        # EXP: Enable initialization.
        # break
        if preprocessing_time_limit > 0 and time() - t0 > preprocessing_time_limit:
            break
        X_train_train, _, y_train_train, _ = train_test_split(
            X_train, y_train, test_size=0.2, random_state=i)
        dt = tree.DecisionTreeClassifier(
            max_depth=experiment_max_depth, random_state=99)
        clf_dt = dt.fit(X_train_train, y_train_train)

        n_nodes = clf_dt.tree_.node_count
        children_left = clf_dt.tree_.children_left
        children_right = clf_dt.tree_.children_right
        feature = clf_dt.tree_.feature
        threshold = clf_dt.tree_.threshold
        targets = clf_dt.tree_.value
        is_leaves = get_is_leaves(n_nodes, children_left, children_right)
        # all_splits = get_all_splits(
        #     n_nodes, feature, threshold, is_leaves)
        # combined_splits = merge_all_splits(combined_splits, all_splits)
        # splits = get_splits_list(combined_splits)
        tree_nodes = get_all_nodes(
            children_left, children_right, n_nodes,
            feature, threshold, is_leaves, combined_splits)
        # combined_nodes = merge_all_nodes(combined_nodes, all_nodes)
        # combined_nodes = merge_all_nodes_last_split(combined_nodes, all_nodes)
        for leaf in leaves:
            path = get_path_for_leaf(
                leaf, tree_nodes, experiment_max_depth, splits,
                X_train,
                y_train)
            if path == None:
                continue
            found = False
            for p in paths:
                if path.is_same_as(p):
                    found = True
                    break
            if not found:
                path.id = len(paths)
                paths.append(path)

    # splits = get_splits_list(combined_splits)
    # nodes = get_nodes_list(combined_nodes)
    targets = np.unique(y_train)  # y_train.unique()
    splits = add_rows_to_splits(splits, X_train)
    # EXP: Set default aggressive mode to True/False.
    use_aggressive_mode = is_large_dataset
    if preprocessing_time_limit > 0 and (time() - t0) > preprocessing_time_limit:
        use_aggressive_mode = False
    data_rows = preprocess(nodes, splits, X_train,
                           y_train, experiment_max_depth,
                           aggressive=use_aggressive_mode,
                           time_limit=preprocessing_time_limit - (time() - t0))
    t2 = time()
    preprocessing_time = t2 - t0
    # print("Total preprocessing time: ", t2 - t0)
    time_limit -= preprocessing_time

    t0 = time()
    clf = DTreeClassifier(paths.copy(), leaves.copy(), nodes.copy(),
                          splits.copy(),
                          tree_depth=experiment_max_depth,
                          targets=targets.copy(),
                          data_rows=data_rows.copy(),
                          max_iterations=max_iterations,
                          time_limit=time_limit,
                          num_master_cuts_round=10,
                          master_beta_constraints_as_cuts=True,
                          master_generate_cuts=False,
                          use_old_sp=use_old_sp,
                          master_solver_type='glop')
    clf.fit(X_train, y_train)

    return clf

