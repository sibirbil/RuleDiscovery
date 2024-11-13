import numpy as np
import pandas as pd
import Datasets as DS
import sys
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/03_RuleDiscovery/github/RuleDiscovery/col_gen_estimator/col_gen_estimator')
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/03_RuleDiscovery/github/RuleDiscovery/col_gen_estimator/examples')
sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/03_RuleDiscovery/github/RuleDiscovery/num_exp/helpers')
from grid_search_helpers import *
from DT_heuristic_helpers import *
from time import time
import math
import random
import pandas as pd
import numpy as np
import getopt
import os
import sys
import csv
import ortools

from sklearn.model_selection import train_test_split

from sklearn import tree


# from col_gen_estimator import DTreeClassifier
# from col_gen_estimator import Split
# from col_gen_estimator import Node
# from col_gen_estimator import Leaf
# from col_gen_estimator import Path
# from col_gen_estimator import Row

from _col_gen_classifier import *
from _bdr_classifier import *
from _dtree_classifier import *
from _parameter_tuner import *
from dtree_experiment import get_is_leaves, get_all_splits, merge_all_splits, get_all_nodes, merge_all_nodes, get_splits_list, \
    get_all_leaves_paths, get_paths_list, get_leaves_list, get_nodes_list, add_rows_to_splits, get_path_for_leaf, preprocess

model = 'DTheuristic'
write = False
numCV: int = 5
testSize = 0.2
randomState = 21
binary = False

problem = DS.banknote

DT_heuristic_pgrid = {'max_depth': [3, 5, 10]}

class Results:
    def __init__(self) -> None:
        self.name = ""
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.preprocessing_time = 0.0
        self.training_time = 0.0
        self.master_time = 0.0
        self.mster_cuts_time = 0.0
        self.master_num_cuts = 0
        self.sp_time_heuristic = 0.0
        self.sp_time_ip = 0.0
        self.sp_cols_added_heuristic = 0
        self.sp_cols_added_ip = 0
        self.col_add_time = 0.0
        self.master_ip_time = 0.0
        self.total_iterations = 0
        self.num_improving_iter = 0
        self.heuristic_hit_rate = 0.0

# ---
save_path = None
numSplits = numCV
target = 'y'
REDUCED_SPLITS = False
time_limit = 300
max_iterations = -1

if save_path is None:
    save_path = f'./results_w_{model}/'

pname = problem.__name__.upper()
print(f'---------{pname}---------')
# get data using prep_data()
X_train, X_test, y_train, y_test = prep_data(problem, randomState=randomState, testSize=testSize,
                                             target=target, model=model, use_binary=binary)

# get all combinations of hyperparameters using get_param_grid
# param_grid_list = get_param_grid(pgrid)
# print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(numSplits,
#                                                                                 len(param_grid_list),
#                                                                                 numSplits * len(param_grid_list)))
# print('---')
#
# evaluation_parameters = pd.DataFrame() # cv performance will be saved here

experiment_max_depth = 2
use_old_sp = False
preprocessing_time_limit = 100
random.seed(10)
train_file = ""
test_file = ""
# sep = ','
# header = 0
results_dir = "./results/"

is_large_dataset = False
if X_train.shape[0] >= 5000:
    is_large_dataset = True

t0 = time()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

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
test_accuracy = clf_dt.score(X_test, y_test)

print("Train Acurracy: ", train_accuracy)
print("Test Acurracy: ", test_accuracy)
t1 = time()
print("time elapsed: ", t1 - d_tree_start_time)

cart_results = Results()
cart_results.name = "CART"
cart_results.train_accuracy = train_accuracy
cart_results.test_accuracy = test_accuracy
cart_results.training_time = t1 - d_tree_start_time

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
print("Total preprocessing time: ", t2 - t0)
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

print('-----')
print('done training')
print('-----')
# print(clf)


print(clf.mp_optimal_, clf.iter_, clf.time_elapsed_)
train_accuracy = clf.score(X_train, y_train)
print("Train Accuracy: ", train_accuracy)
test_accuracy = clf.score(X_test, y_test)
print("Test Accuracy: ", test_accuracy)
t1 = time()
print("time elapsed: ", t1-t0)

cg1_results = Results()
cg1_results.name = "CG1"
cg1_results.train_accuracy = train_accuracy
cg1_results.test_accuracy = test_accuracy
cg1_results.preprocessing_time = preprocessing_time
cg1_results.training_time = clf.time_elapsed_
cg1_results.master_time = clf.time_spent_master_
cg1_results.mster_cuts_time = clf.master_problem.cut_gen_time
cg1_results.master_num_cuts = clf.master_problem.total_cuts_added
cg1_results.sp_time_heuristic = clf.time_spent_sp_[0][0]
cg1_results.sp_time_ip = sum(clf.time_spent_sp_[1])
cg1_results.sp_cols_added_heuristic = clf.num_col_added_sp_[0][0]
cg1_results.sp_cols_added_ip = sum(clf.num_col_added_sp_[1])
cg1_results.master_ip_time = clf.time_spent_master_ip_
cg1_results.total_iterations = clf.iter_
cg1_results.num_improving_iter = clf.num_improving_iter_
cg1_results.col_add_time = clf.time_add_col_
failed_heuristic_rounds = clf.subproblems[0][0].failed_rounds
successful_heuristic_rounds = clf.subproblems[0][0].success_rounds
cg1_results.heuristic_hit_rate = 0.0
if (successful_heuristic_rounds + failed_heuristic_rounds) > 0:
    cg1_results.heuristic_hit_rate = float(successful_heuristic_rounds) / \
        (successful_heuristic_rounds + failed_heuristic_rounds)

added_rows = []
for r in range(len(data_rows)):
    if clf.master_problem.added_row[r]:
        added_rows.append(r)

print("Total added rows: ", len(added_rows))
print("Last reset iter = ", clf.master_problem.last_reset_iter_)

for sp in clf.subproblems[1]:
    print("Solve times: ", sp.solve_times_)
    print("Gaps: ", sp.gaps_)
    for key in sp.tunable_params_:
        sp.tunable_params_[key].print_stats()

attrs = vars(cg1_results)
print('\n'.join("%s: %s" % item for item in attrs.items()))

train_file_name = os.path.basename(train_file)

prefix = results_dir + 'T_' + train_file_name + \
    '_d_' + str(experiment_max_depth)
results_filename = prefix + '_results.csv'
with open(results_filename, 'w', newline='') as f:
    # fieldnames lists the headers for the csv.
    w = csv.DictWriter(f, fieldnames=sorted(vars(cart_results)))
    w.writeheader()
    w.writerow({k: getattr(cart_results, k)
                for k in vars(cart_results)})
    w.writerow({k: getattr(cg1_results, k)
                for k in vars(cg1_results)})

print(f'Nr of rules: {len(clf.leaves)}')
print(f'Average rule length: {DT_heuristic_avg_rule_length(clf)}')
print(f'Average rule length per sample: {DT_heuristic_avg_rule_length_sample(clf,X_test)}')