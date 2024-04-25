"""
We need to be able to access the code of other repo, specifically:
- ./binoct      # for BinOCT
- ./pydl8.5-lbguess     # for FSDT
- ./FairCG/src      # for FairCG and CG
"""
import sys
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/pydl8.5-lbguess')
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/FairCG/src')
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/binoct')

import os
# import Datasets_binary as DS
from ruxg import RUGClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import time

# # for FSDT
import FSDT_helpers as FSDT_helpers
from dl85 import DL85Classifier

# # for BinOCT
import learn_class_bin as lcb

# # for CG and FairCG
from CG_helpers import *

# For FairRUG
import fairconstraints as FC


def prep_data(problem, randomState = 0, testSize = 0.2, target = 'y', model = 'RUG', use_binary=False):
    """
    Preps the data into train/test set.

    Parameters:
    problem (DS.data): function of DS indicating which dataset to use (e.g. DS.hearts)
    randomState (int): seed (default 0)
    testSize (float): size of test set (default 0.3)
    target (str): name of target feature (default 'y')
    model (str): name of the model, any of 'RUG', 'FSDT', 'binoct', 'CG', 'FairCG', 'FairRUG' (default 'RUG')
    use_binary (bool): indicating whether data should be put into bins (categorized) and then binarized. Default False

    Returns:
    A numpy array of the train data (X_train and y_train).
    A numpy array of the test data (X_test and y_test).
    """
    pname = problem.__name__.upper()

    # FSDT, CG, and FairCG can only handle binary data
    if model=='FSDT' or model=='CG' or model=='FairCG':
        use_binary=True

    if use_binary:
        print('Binary data is used.')
        try: df = pd.read_csv(f'../datasets/binary/{pname}_binary.csv')
        except: df = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
    else:
        print('Original data is used with one-hot encoding for categorical variables.')
        try: df = problem('../datasets/original/')
        except: df = problem('./datasets/original/')

    # if model == 'CG':
    #     df = pd.read_csv(f'../datasets/CG_binarized/{pname.lower()}_binary.csv')

    df = df.dropna(subset=[target])
    y = np.array(df[target]).astype(int)
    df = df.drop(target, axis=1)
    X = np.array(df)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=testSize, stratify=y)

    return X_train, X_test, y_train, y_test


# def save_data_splits(pname, X, y, numSplits=5, shuffle=True, randomState=21, data_path='./prepped_data/'):
#     kf = KFold(n_splits=numSplits, shuffle=True, random_state=randomState)
#     foldnum = 0
#     for train_index, val_index in kf.split(X):
#         foldnum += 1
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
#
#         train = pd.DataFrame(X_train)
#         train.columns = ['X_' + str(i) for i in range(len(train.columns))]
#         train['y'] = y_train
#         train.to_csv(f'{data_path}{pname}_train{foldnum}.csv', index=False, sep=';')
#
#         val = pd.DataFrame(X_val)
#         val.columns = ['X_' + str(i) for i in range(len(val.columns))]
#         val['y'] = y_val
#         val.to_csv(f'{data_path}{pname}_val{foldnum}.csv', index=False, sep=';')
#
#     return


def get_param_grid(param_grid):
    """
    Parameters
    ----------
    param_grid (dict): parameter grid in the form that the key is the parameter, and the value is a list of values for that parameter

    Returns
    -------
    a list of dictionaries, where each item is a dictionary that has only one value for each key
    -- as many items in the list as there are combinations of the parameter values
    """
    list = []
    for k in param_grid:
        list.append(param_grid[k])

    keys, values = zip(*param_grid.items())
    combinations = [p for p in itertools.product(*list)]
    return [dict(zip(keys, v)) for v in combinations]


def cv(param_grid, X, y, pname, numSplits = 5, randomState = 0, model = 'RUG',
       fairness_metric = None, RUG_rule_length_cost=False, RUG_threshold=None):
    """
    Preps the data into train/test set.

    Parameters:
    param_grid (dict): dictionary with parameter grid
    X (np array): X_train from prep_data()
    y (np array): y_train from prep_data()
    numSplits (int): default 5
    randomState (int): seed (default 0)

    Returns:
    param_grid_out (DataFrame): DataFrame with parameters and average accuracy over the cross-validation
    """

    # Split data into train and validation: random state fixed for reproducibility
    kf = KFold(n_splits=numSplits,shuffle=True,random_state=randomState)
    accuracy = [] # initialize list where we will save the accuracy score of each fold

    if model == 'RUG':
        print(f'{model} {numSplits}-fold cross validation with max_depth={param_grid["max_depth"]}, '
              f'pen_par={param_grid["pen_par"]}, max_RMP_calls={param_grid["max_RMP_calls"]}, '
              f'and rule_length_cost={RUG_rule_length_cost}')
        # kf-fold cross-validation loop
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            clf = None
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            if RUG_threshold is None:
                clf = RUGClassifier(random_state=randomState, rule_length_cost=RUG_rule_length_cost)
            else:
                clf = RUGClassifier(random_state=randomState, threshold=RUG_threshold, rule_length_cost=RUG_rule_length_cost)
            clf.set_params(**param_grid)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'FSDT':
        print(f'{model} {numSplits}-fold cross validation with max_depth={param_grid["max_depth"]}')
        # kf-fold cross-validation loop
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            clf = DL85Classifier(time_limit=300, desc=True)
            clf.set_params(**param_grid)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'binoct':
        print(f'{model} {numSplits}-fold cross validation with max_depth={param_grid["max_depth"]}')
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            print(pname)
            # filename = f'{data_path}{pname}_train{foldnum}.csv'
            filename = f'../datasets/train-test-splits/binary/{pname}_train{foldnum}.csv'

            lcb.main(["-f", filename, "-d", param_grid['max_depth'], "-t", 300, "-p", 100])

            import sys
            sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/03_RuleDiscovery/github/RuleDiscovery/num_exp')
            import predict_file
            # y_val = pd.read_csv(f'./prepped_data_binoct/{pname}_val{foldnum}.csv', sep=';')['y']
            y_val = pd.read_csv(f'../datasets/train-test-splits/binary/{pname}_val{foldnum}.csv', sep=';')['y']
            # y_pred, path_lengths = predict_file.main([f'{data_path}{pname}_val{foldnum}.csv'])
            y_pred, path_lengths = predict_file.main([f'../datasets/train-test-splits/binary/{pname}_val{foldnum}.csv'])

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'CG':
        CG_EqOfOp = []
        CG_HammingEqOdd = []

        print(f'{model} {numSplits}-fold cross validation with complexity={param_grid["complexity"]} and '
              f'epsilon={param_grid["epsilon"]}')
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            res, classif = run_CG(pname, X_train, X_val, y_train, y_val, param_grid)

            accuracy.append(res.res['accuracy'])
            CG_EqOfOp.append(res.res['EqualOpportunity'])
            CG_HammingEqOdd.append(res.res['EqualizedOdds'])
    elif model == 'FairRUG':
        if fairness_metric is None:
            print('Please provide a fairness metric.')
            return
        print(
            f'{model} {numSplits}-fold cross validation with max_depth={param_grid["max_depth"]}, '
            f'pen_par={param_grid["pen_par"]}, max_RMP_calls={param_grid["max_RMP_calls"]}, '
            f'fair_eps={param_grid["fair_eps"]}, fairness_metric={fairness_metric}, and '
            f'rule_length_cost={RUG_rule_length_cost}')

        # Obtain classes and groups
        groups = pd.unique(X[:, 0])
        groups.sort()
        classes = pd.unique(y)
        classes.sort()
        unfairness=[] # initialize list to save unfairness values of each fold

        # kf-fold cross-validation loop
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            clf = None
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # For each pair of groups, create sets P (list of vectors/np.array)
            constraintSetPairs_train, pairs = FC.create_setsPI(X_train, y_train, groups, metric=fairness_metric)
            constraintSetPairs_test, pairs = FC.create_setsPI(X_val, y_val, groups, metric=fairness_metric)

            if RUG_threshold is None:
                clf = RUGClassifier(random_state=randomState, fair_metric=fairness_metric,
                                    rule_length_cost=RUG_rule_length_cost)
            else:
                clf = RUGClassifier(random_state=randomState, threshold=RUG_threshold, fair_metric=fairness_metric,
                                    rule_length_cost=RUG_rule_length_cost)
            clf.set_params(**param_grid)
            clf.fit(X_train, y_train, groups=constraintSetPairs_train)
            y_pred = clf.predict(X_val)

            # RUG_unfairness = FC.fairnessEvaluation(y_val, y_pred, constraintSetPairs_test, classes, pairs)
            if len(classes) > 2:
                RUG_unfairness = FC.fairnessEvaluation(y_val, y_pred, constraintSetPairs_test, classes, pairs)
            elif len(classes)==2 and len(groups)==2:
                if fairness_metric=='odm':
                    #RUG_unfairness = FC.binary_odm(y_val, y_pred, constraintSetPairs_test, classes, pairs)
                    RUG_unfairness = FC.fairnessEvaluation(y_val, y_pred, constraintSetPairs_test, classes, pairs)
                if fairness_metric=='EqOpp':
                    RUG_unfairness = FC.binary_EqOpp(y_val, y_pred, constraintSetPairs_test, classes, pairs)
                if fairness_metric=='dmc':
                    RUG_unfairness = FC.binary_EqOdds(y_val, y_pred, constraintSetPairs_test, classes, pairs)
            elif len(classes)==2 and len(groups)>2:
                "here for attrition"
                RUG_unfairness = FC.fairnessEvaluation(y_val, y_pred, constraintSetPairs_test, classes, pairs)

            unfairness.append(RUG_unfairness) # save unfairness values of fold

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'FairCG':
        CG_EqOfOp = []
        CG_HammingEqOdd = []
        if fairness_metric is None:
            print('Please provide a fairness metric.')
            return
        unfairness = [] # initialize list to save unfairness values of each fold

        print(f'{model} {numSplits}-fold cross validation with complexity={param_grid["complexity"]}, '
              f'epsilon={param_grid["epsilon"]}, and fairness_metric={fairness_metric}')
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            res, classif = run_CG(pname, X_train, X_val, y_train, y_val, param_grid, fairness_metric=fairness_metric)

            # save accuracy of fold
            accuracy.append(res.res['accuracy'])
            CG_EqOfOp.append(res.res['EqualOpportunity'])
            CG_HammingEqOdd.append(res.res['EqualizedOdds'])

    else:
        print("WARNING: please specify model type (either 'RUG' or 'FSDT' or 'binoct' or 'CG' or 'FairRUG')")
        return

    print(f'Mean Accuracy for these parameters: {np.mean(accuracy)}')
    print('---')

    param_grid_out = pd.DataFrame(param_grid, index=[0])
    param_grid_out['Accuracy'] = np.mean(accuracy)

    if model=='FairRUG':
        param_grid_out['Unfairness'] = np.mean(unfairness)
    if model=='FairCG':
        if fairness_metric=='EqOfOp':
            # print(CG_EqOfOp)
            # if isinstance(CG_EqOfOp[0], list):
            #     CG_EqOfOp = [item for sublist in CG_EqOfOp for item in sublist]
            # param_grid_out['Unfairness'] = np.mean(CG_EqOfOp)
            param_grid_out['Unfairness'] = np.nanmean(CG_EqOfOp)
        elif fairness_metric=='HammingEqOdd' or fairness_metric=='unfair':
            # print(CG_HammingEqOdd)
            # if isinstance(CG_HammingEqOdd[0], list):
            #     CG_HammingEqOdd = [item for sublist in CG_HammingEqOdd for item in sublist]
            # param_grid_out['Unfairness'] = np.mean(CG_HammingEqOdd)
            param_grid_out['Unfairness'] = np.nanmean(CG_HammingEqOdd)

    return param_grid_out


def find_best_params(evaluation_parameters, model = None):
    """
    Parameters
    ----------
    evaluation_parameters (DataFrame):

    Returns
    -------
    best_params (dict): dictionary with best parameters
    """

    """
    if we are looking at FairRUG or FairCG, then we choose the best model / the best set of parameters 
    based on the unfairness value – the smaller, the better
    for all other methods, we base it on accuracy – the higher, the better
    """
    # if 'Unfairness' in evaluation_parameters.columns:
    print(evaluation_parameters)
    if model == 'FairRUG' or model == 'FairCG':
        print('Best parameters are chosen based on unfairness.')
        evaluation_parameters = evaluation_parameters.drop('Accuracy', axis=1)
        # evaluation_parameters.dropna()
        # idx = evaluation_parameters['Unfairness'].idxmin()
        idx = evaluation_parameters['Unfairness'].idxmin(skipna=True)
        best_params = evaluation_parameters.drop('Unfairness', axis=1).loc[idx]
    else:
        print('Best parameters are chosen based on accuracy.')
        idx = evaluation_parameters['Accuracy'].idxmax()
        best_params = evaluation_parameters.drop('Accuracy', axis=1).loc[idx]

    best_params = dict(best_params)
    for key in best_params:
        best_params[key] = best_params[key].astype(evaluation_parameters.dtypes.loc[key])

    return best_params


def get_results(y_test, y_pred, clf, X_test, model):
    """
    function to get results for FSDT, RUG, or FairRUG

    Parameters
    ----------
    y_test: np array with y_test values
    y_pred: np array with y_pred values
    clf: fitted model
    X_test: test set
    model (str): model type/name

    Returns
    -------
    dictionary with all the scores
    """
    scores = {'Accuracy': [],
              'F1-score':[],
              'MCC':[],
              'Nr of Rules': [],
              'Avg. Rule Length': [],
              'Avg. Nr. Rules per Sample': [],
              'Avg. Rule Length per Sample':[],
              'Fit Time':[]}

    scores['Accuracy'].append(accuracy_score(y_test, y_pred))
    if len(np.unique(y_test))>2:
        scores['F1-score-macro'] = [f1_score(y_test,y_pred, average='macro')]
        scores['F1-score-weighted'] = [f1_score(y_test, y_pred, average='weighted')]
    else:
        scores['F1-score'].append(f1_score(y_test, y_pred))
    scores['MCC'].append(matthews_corrcoef(y_test, y_pred))

    if (model == 'RUG' or model == 'FairRUG'):
        scores['Nr of Rules'].append(clf.get_num_of_rules())
        scores['Avg. Rule Length'].append(clf.get_avg_rule_length())
        scores['Avg. Nr. Rules per Sample'].append(clf.get_avg_num_rules_per_sample())
        scores['Avg. Rule Length per Sample'].append(clf.get_avg_rule_length_per_sample())
        scores['Fit Time'].append(clf.get_fit_time())
    elif model == 'FSDT':
        scores['Nr of Rules'].append(FSDT_helpers.get_num_leaves(clf.tree_))
        scores['Avg. Rule Length'].append(FSDT_helpers.get_avg_rule_length(clf.tree_))
        scores['Avg. Nr. Rules per Sample'].append(1)
        scores['Avg. Rule Length per Sample'].append(FSDT_helpers.get_avg_rule_length_per_sample(clf.tree_, X_test))
        scores['Fit Time'].append(clf.runtime_)

    return scores


def write_results(pname, scores, path, binary, shape, best_params, param_grid, model, fairness_metric=None,
                  RUG_rule_length_cost=None, RUG_threshold=None):
    """
    write results to a .txt file

    Parameters
    ----------
    pname (str): name of dataset
    scores (dict): dictionary with scores, obtained from get_results()
    path (str): save path folder
    binary (bool): binary True or False
    shape: shape of the entire dataset (train & test)
    best_params (dict): dictionary with the best parameters, obtained by find_best_params()
    param_grid (dict): parameter grid used for grid search
    model (str): model name/type
    fairness_metric (str): fairness metric parameter

    Returns
    -------
    """
    fname: str = path + model + '_cv_'

    if not os.path.exists(path):
        os.makedirs(path)

    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print(model, file=f)
        if model == 'FairRUG' or model=='RUG':
            print(f'Rule length cost: {RUG_rule_length_cost}', file=f)
            print(f'Threshold rule weights: {RUG_threshold}', file=f)
        if model == 'FairRUG' or model=='FairCG':
            print(f'Fairness metric used: {fairness_metric}', file=f)
        print(f'Binarization used: {binary}', file=f)
        print(f'Dataset shape (train+test): {shape} \n', file=f)
        print(f'Parameters tried for grid search: \n {param_grid} \n', file=f)
        print(f'Best parameters: \n {best_params} \n' ,file=f)
        for method in scores.keys():
            txt = '{0}: \t {1:.4f} ({2:.4f})'.format(method,
                                                     np.mean(scores[method]), np.std(scores[method]))

            print(txt, file=f)

        print('<---\n', file=f)

    return


def run(problem, pgrid, save_path = None,
        randomState = 0, testSize=0.2, numSplits=5, binary = True, write=True,
        model = 'RUG', target = 'y',
        fairness_metric=None, RUG_rule_length_cost=False, RUG_threshold=None,
        RUG_record_fairness=False):

    if save_path is None:
        save_path = f'./results_w_{model}/'

    pname = problem.__name__.upper()
    print(f'---------{pname}---------')
    # get data using prep_data()
    X_train, X_test, y_train, y_test = prep_data(problem, randomState=randomState, testSize=testSize,
                                                 target=target, model=model, use_binary=binary)

    # get all combinations of hyperparameters using get_param_grid
    param_grid_list = get_param_grid(pgrid)
    print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(numSplits,
                                                                                    len(param_grid_list),
                                                                                    numSplits*len(param_grid_list)))
    print('---')

    evaluation_parameters = pd.DataFrame() # cv performance will be saved here
    # loop through param_grid_list
    for param_grid in param_grid_list:
        # for each parameter combination, run k-fold cv
        param_grid_out = cv(param_grid, X_train, y_train, pname=pname, numSplits=numSplits, randomState=randomState,
                            model=model, fairness_metric=fairness_metric,
                            RUG_rule_length_cost=RUG_rule_length_cost, RUG_threshold=RUG_threshold)
        # save accuracy result of cv in dataframe
        evaluation_parameters = pd.concat([evaluation_parameters, param_grid_out], ignore_index=True)

    # find best parameters based on average accuracy
    best_params = find_best_params(evaluation_parameters, model=model)
    print(f'Best parameters: {best_params}')
    # make sure that parameters have correct data type
    for k in best_params:
        if isinstance(best_params[k], np.integer):
            best_params[k] = int(best_params[k])

    # test model on hold-out test set (X_test)
    if model == 'RUG':
        if RUG_threshold is None:
            clf_final = RUGClassifier(random_state=randomState, rule_length_cost=RUG_rule_length_cost)
        else:
            clf_final = RUGClassifier(random_state=randomState, threshold=RUG_threshold, rule_length_cost=RUG_rule_length_cost)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train)
        y_pred = clf_final.predict(X_test)
        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)

        ## calculate fairness scores for RUG
        if RUG_record_fairness:
            # # Obtain classes and groups
            groups = pd.unique(X_train[:, 0])
            groups.sort()
            classes = pd.unique(y_train)
            classes.sort()

            # # For each pair of groups, create sets P (list of vectors/np.array)
            if len(classes) == 2:
                print('-------DMC------')
                constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric='dmc')
                RUG_EqualizedOdds = FC.binary_EqOdds(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness DMC'] = [1 - RUG_EqualizedOdds]

                print('-------EqOpp------')
                constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric='EqOpp')
                RUG_EqualOpportunity = FC.binary_EqOpp(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness Equal Opportunity'] = [1 - RUG_EqualOpportunity]

                print('-------ODM------')
                constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric='odm')
                RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness ODM'] = [1 - RUG_unfairness]

            else:
                # dmc
                constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric='dmc')
                RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness DMC'] = [1 - RUG_unfairness]

                # odm
                constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric='odm')
                RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness ODM'] = [1 - RUG_unfairness]

    elif model == 'FSDT':
        clf_final = DL85Classifier(time_limit=300, desc=True)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train)
        y_pred = clf_final.predict(X_test)
        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)
    elif model == 'binoct':
        # filename = f'{data_path}{pname}_train_complete.csv'
        # if binary:
        filename = f'../datasets/train-test-splits/binary/{pname}_train_complete.csv'
        startTime = time.time()
        lcb.main(["-f", filename, "-d", best_params['max_depth'], "-t", 300, "-p", 100])
        endTime = time.time()
        predictTime = endTime - startTime

        import predict_file
        # y_pred, path_lengths = predict_file.main([f'./prepped_data/{pname}_test.csv'])
        # if binary:
        y_pred, path_lengths = predict_file.main([f'../datasets/train-test-splits/binary/{pname}_test.csv'])
        print(path_lengths)

        import inspect
        t = inspect.getsource(predict_file.predict)
        scores = {}
        scores['Accuracy'] = [accuracy_score(y_test, y_pred)]
        scores['F1'] = [f1_score(y_test, y_pred)]
        scores['MCC'] = [matthews_corrcoef(y_test, y_pred)]
        scores['Nr of Rules'] = [t.count('return')]
        scores['Avg. Rule Length'] = [best_params['max_depth']]
        scores['Avg. Nr Rules per Sample'] = [1]
        scores['Avg. Rule Length per Sample'] = [np.mean(path_lengths)]
        scores['Fit Time'] = [predictTime]
    elif model == 'CG':
        res, classif = run_CG(pname, X_train, X_test, y_train, y_test, best_params)

        final_rule_set = classif.fitRuleSet

        preds = classif.predict(X_test)
        scores = {}
        scores['Accuracy'] = [accuracy_score(y_test, preds)]
        scores['F1'] = [f1_score(y_test, preds)]
        scores['MCC'] = [matthews_corrcoef(y_test, preds)]
        scores['fitRuleSet - Nr of Rules'] = len(final_rule_set)
        scores['fitRuleSet - Avg. Rule Length'] = np.mean(np.sum(final_rule_set, axis=1))
        scores['fitRuleSet - Avg. Nr Rules per Sample'], scores['fitRuleSet - Avg. Rule Length per Sample'] = CG_rules_per_sample(X_test, final_rule_set)
        scores['Fit Time'] = res.res['times']
        scores['Fairness DMC'] = [1-value for value in res.res['EqualizedOdds']]
        scores['Fairness EqualOpportunity'] = [1-value for value in res.res['EqualOpportunity']]
        scores['Fairness ODM'] = [1-value for value in res.res['ODM']]
    elif model == 'FairRUG':
        # Obtain classes and groups
        groups = pd.unique(X_train[:, 0])
        groups.sort()
        classes = pd.unique(y_train)
        classes.sort()

        # For each pair of groups, create sets P (list of vectors/np.array)
        constraintSetPairs_train, pairs = FC.create_setsPI(X_train, y_train, groups, metric=fairness_metric)
        constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric=fairness_metric)

        if RUG_threshold is None:
            clf_final = RUGClassifier(random_state=randomState, fair_metric=fairness_metric, rule_length_cost=RUG_rule_length_cost)
        else:
            clf_final = RUGClassifier(random_state=randomState, threshold=RUG_threshold,
                                      fair_metric=fairness_metric, rule_length_cost=RUG_rule_length_cost)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train, groups=constraintSetPairs_train)
        y_pred = clf_final.predict(X_test)

        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)

        if len(classes) > 2:
            RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
            if fairness_metric == 'dmc':
                scores['Fairness DMC'] = [1 - RUG_unfairness]
            else:
                scores['Fairness ODM'] = [1 - RUG_unfairness]

        elif len(classes) == 2 and len(groups)==2:
            if fairness_metric == 'dmc':
                RUG_EqualizedOdds = FC.binary_EqOdds(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness DMC'] = [1 - RUG_EqualizedOdds]

            if fairness_metric == 'EqOpp':
                RUG_EqualOpportunity = FC.binary_EqOpp(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness Equal Opportunity'] = [1 - RUG_EqualOpportunity]

            if fairness_metric == 'odm':
                # RUG_unfairness = FC.binary_odm(y_test, y_pred, constraintSetPairs_train, classes, pairs)
                RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                scores['Fairness ODM'] = [1 - RUG_unfairness]
        elif len(classes)==2 and len(groups)>2:
                "here for attrition3"
                RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
                if fairness_metric == 'dmc':
                    scores['Fairness DMC'] = [1 - RUG_unfairness]
                else:
                    scores['Fairness ODM'] = [1 - RUG_unfairness]

            

    elif model == 'FairCG':
        res, classif = run_CG(pname, X_train, X_test, y_train, y_test, best_params, fairness_metric=fairness_metric)

        final_rule_set = classif.fitRuleSet

        preds = classif.predict(X_test)
        scores = {}
        scores['Accuracy'] = [accuracy_score(y_test, preds)]
        scores['F1'] = [f1_score(y_test, preds)]
        scores['MCC'] = [matthews_corrcoef(y_test, preds)]
        scores['fitRuleSet - Nr of Rules'] = len(final_rule_set)
        scores['fitRuleSet - Avg. Rule Length'] = np.mean(np.sum(final_rule_set, axis=1))
        scores['fitRuleSet - Avg. Nr Rules per Sample'], scores['fitRuleSet - Avg. Rule Length per Sample'] = CG_rules_per_sample(X_test, final_rule_set)
        scores['Fit Time'] = res.res['times']
        scores['Fairness DMC'] = [1-value for value in res.res['EqualizedOdds']]
        scores['Fairness EqualOpportunity'] = [1-value for value in res.res['EqualOpportunity']]
        scores['Fairness ODM'] = [1-value for value in res.res['ODM']]

    else:
        return

    if write:
        shape = (len(X_train)+len(X_test), X_train.shape[1]+1)
        write_results(pname, scores, path = save_path, binary = binary,
                      shape = shape, best_params=best_params,
                      param_grid = pgrid, model=model,
                      fairness_metric=fairness_metric,
                      RUG_rule_length_cost=RUG_rule_length_cost,
                      RUG_threshold=RUG_threshold)

    return

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# helper functions for traditional ML models
# used in case study & multiclass classification comparison (Table 2 manuscript)

# ----------------------
# sklearn tree and tree ensembles helper functions
# ----------------------

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

# ----------------------
# LightGBM helper functions
# ----------------------
# functions to get average rule length over all trees in the model
def light_gbm_leaf_depth(tree_structure, current_depth=0, leaf_depths=None):
    if leaf_depths is None:
        leaf_depths = {}

    # Check if it's a leaf node
    if 'leaf_index' in tree_structure:
        leaf_depths[tree_structure['leaf_index']] = current_depth
    else:
        # If not a leaf, traverse left and right children
        left_child = tree_structure.get('left_child')
        right_child = tree_structure.get('right_child')

        if left_child:
            light_gbm_leaf_depth(left_child, current_depth + 1, leaf_depths)
        if right_child:
            light_gbm_leaf_depth(right_child, current_depth + 1, leaf_depths)

    return leaf_depths


def light_gbm_avg_rule_length(clf):
  path_lengths = []
  for tree in clf._Booster.dump_model()['tree_info']:
    tree_structure = tree['tree_structure']

    leaf_depths = light_gbm_leaf_depth(tree_structure)

    path_lengths.extend(leaf_depths.values())

  return np.mean(path_lengths)

# functions to get average rule length for each sample in X_test over all trees in the model

def light_gbm_path_length_for_sample(tree_structure, sample, current_depth=0):
    # Check if it's a leaf node
    if 'leaf_index' in tree_structure:
        return current_depth

    # Extract the split feature index and threshold
    if 'split_feature' in tree_structure:
      split_feature = tree_structure['split_feature']
      threshold = tree_structure['threshold']
    else:
      return 0

    # Check which child node to traverse
    if sample[split_feature] <= threshold:
        child_node = tree_structure['left_child']
    else:
        child_node = tree_structure['right_child']

    # Recursively traverse the tree
    return light_gbm_path_length_for_sample(child_node, sample, current_depth + 1)


def light_gbm_avg_rule_length_per_sample(clf, X_test):

  # Record path lengths for each sample in X_test
  path_lengths = {}

  # for each sample, record the length of the path used to classify it in each tree
  for tree in clf._Booster.dump_model()['tree_info']:
    tree_structure = tree['tree_structure']
    # print(f"Tree: {tree['tree_index']}")

    for idx, sample in enumerate(X_test):
      path = light_gbm_path_length_for_sample(tree_structure, sample)
      if idx in path_lengths:
        path_lengths[idx].append(path)
      else:
        path_lengths[idx] = [path]

  # record average rule length for each sample
  avg_length = []
  for idx,values in path_lengths.items():
    avg_length.append(np.mean(values))


  # return the average of the average rule length per sample
  return np.mean(avg_length)