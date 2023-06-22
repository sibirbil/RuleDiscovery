"""
We need to be able to access the code of other repo, specifically:
- ./binoct      # for BinOCT
- ./pydl8.5-lbguess     # for FSDT
- ./FairCG/src      # for FairCG and CG
- ./FairCG/sample_experiment_notebooks      # for FairCG and CG
"""
import sys
# sys.path.insert(1,'...')

# sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/binoct')
# sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/pydl8.5-lbguess')
# sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/FairCG/src')
# sys.path.insert(1, '/Users/tabearober/OneDrive - UvA/Interpretable ML/13_MPinXAI/Code/FairCG/sample experiment notebooks')


import os
import Datasets as DS
from ruxg import RUGClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import time

# for FSDT
import DL85_helpers as FastSDTOpt_helpers
from dl85 import DL85Classifier

# for BinOCT
import learn_class_bin as lcb

# for CG and FairCG
# from test_helpers import *
from CG_helpers import *
import fairconstraints as FC


def prep_data(problem, binary=True, randomState = 0, testSize = 0.3, target = 'y', numSplits=5,
              save_splits=True, data_path='./prepped_data/'):
    """
    Preps the data into train/test set.

    Parameters:
    problem (DS.data): function of DS indicating which dataset to use (e.g. DS.hearts)
    binary (bool): indicating whether data should be put into bins (categorized) and then binarized. Default binary = True
    randomState (int): seed (default 0)
    testSize (float): size of test set (default 0.3)
    target (str): name of target feature (default 'y')
    numSplits (int): k-fold cross validation
    save_splits (bool): should splits of the data be saved as .csv files? (default True)
    data_path (str): location where splits should be saved

    Returns:
    A numpy array of the train data (X_train and y_train).
    A numpy array of the test data (X_test and y_test).
    """
    pname = problem.__name__.upper()

    if binary:
        df = problem('./datasets/')
        df_c = pd.DataFrame()
        for column in df.columns:
            if len(df[column].unique()) > 2:
                df_c[column] = pd.cut(df[column], 5, labels=[1, 2, 3, 4, 5])
            else:
                df_c[column] = df[column]

        df_binary = pd.get_dummies(df_c, columns=df.columns.drop(target), drop_first=True)
        X = df_binary.drop(target, axis=1)

        y = df_binary[target]
        X = np.array(X)
        y = np.array(y)
        print(f'Dataset shape: {df_binary.shape}')
        # print('---')
        if df_binary.shape[1] > 1000:
            print('Binary dataset too big')
            return
    else:
        df = problem('./datasets/')
        X = np.array(df.drop(target, axis=1))
        y = np.array(df[target])

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=testSize, stratify=y)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save train and test set as .csv files
    train = pd.DataFrame(X_train)
    train.columns = ['X_' + str(i) for i in range(len(train.columns))]
    train['y'] = y_train
    train.to_csv(f"{data_path}{pname}_train_complete.csv", index=False, sep=';')

    test = pd.DataFrame(X_test)
    test.columns = ['X_' + str(i) for i in range(len(test.columns))]
    test['y'] = y_test
    test.to_csv(f"{data_path}{pname}_test.csv", index=False, sep=';')

    if save_splits:
        save_data_splits(pname, X_train, y_train, numSplits=numSplits, randomState=randomState, data_path=data_path)

    return X_train, X_test, y_train, y_test


def save_data_splits(pname, X, y, numSplits=5, shuffle=True, randomState=21, data_path='./prepped_data/'):
    kf = KFold(n_splits=numSplits, shuffle=True, random_state=randomState)
    foldnum = 0
    for train_index, val_index in kf.split(X):
        foldnum += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train = pd.DataFrame(X_train)
        train.columns = ['X_' + str(i) for i in range(len(train.columns))]
        train['y'] = y_train
        train.to_csv(f'{data_path}{pname}_train{foldnum}.csv', index=False, sep=';')

        val = pd.DataFrame(X_val)
        val.columns = ['X_' + str(i) for i in range(len(val.columns))]
        val['y'] = y_val
        val.to_csv(f'{data_path}{pname}_val{foldnum}.csv', index=False, sep=';')

    return


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


def cv(param_grid, X, y, pname, numSplits = 5, randomState = 0, model = 'RUG', data_path='./prepped_data/',
       fairness_metric = None):
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
        print(f'{model} {numSplits}-fold cross validation with max_depth={param_grid["max_depth"]}, pen_par={param_grid["pen_par"]}, max_RMP_calls={param_grid["max_RMP_calls"]}')
        # kf-fold cross-validation loop
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            clf = None
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # clf = RUGClassifier(random_state=randomState)
            clf = RUGClassifier(random_state=randomState, threshold=0.05, rule_length_cost=True)
            clf.set_params(**param_grid)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'FastSDTOpt':
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
            filename = f'{data_path}{pname}_train{foldnum}.csv'

            lcb.main(["-f", filename, "-d", param_grid['max_depth'], "-t", 300, "-p", 100])

            import sys
            sys.path.insert(1, data_path)
            import predict_file
            y_val = pd.read_csv(f'./prepped_data_binoct/{pname}_val{foldnum}.csv', sep=';')['y']
            y_pred, path_lengths = predict_file.main([f'{data_path}{pname}_val{foldnum}.csv'])

            # save accuracy of fold
            accuracy.append(accuracy_score(y_val, y_pred))
    elif model == 'CG':
        CG_EqOfOp = []
        CG_HammingEqOdd = []
        test_params = {
            'price_limit': 45,
            'train_limit': 300,
            'fixed_model_params': {
                'ruleGenerator': 'Hybrid',
                'masterSolver': 'barrierCrossover',
                'numRulesToReturn': 100,
                'fairness_module': 'unfair'
            },
        }
        print(f'{model} {numSplits}-fold cross validation with complexity={param_grid["complexity"]}')
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            train = pd.DataFrame(X_train)
            train.columns = ['X_' + str(i) for i in range(len(train.columns))]
            train['y'] = y_train

            val = pd.DataFrame(X_val)
            val.columns = ['X_' + str(i) for i in range(len(val.columns))]
            val['y'] = y_val

            train = train.astype(bool)
            val = val.astype(bool)

            # Set up reporting
            eps = 1
            res = TestResults(pname + ' ' + '(%d,%d)' % (eps, param_grid['complexity']) + '-' + str(foldnum))
            res.res['eps'] = eps
            res.res['C'] = param_grid['complexity']

            # Set hyperparameters
            test_params = test_params.copy()
            test_params['fixed_model_params']['epsilon'] = eps
            test_params['fixed_model_params']['ruleComplexity'] = param_grid['complexity']

            # Run CG
            saved_rules = None
            res, classif = runSingleTest(train.drop('y', axis=1).to_numpy(), train['y'].to_numpy(),
                                         train['X_0'].to_numpy(),
                                         val.drop('y', axis=1).to_numpy(), val['y'].to_numpy(),
                                         val['X_0'].to_numpy(),
                                         test_params,
                                         saved_rules, res, colGen=True, rule_filter=False)

            # save accuracy of fold
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
            f'fair_eps={param_grid["fair_eps"]}, fairness_metric={fairness_metric}')

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

            # clf = RUGClassifier(random_state=randomState)
            clf = RUGClassifier(random_state=randomState, threshold=0.05, fair_metric=fairness_metric)
            clf.set_params(**param_grid)
            clf.fit(X_train, y_train, groups=constraintSetPairs_train)
            y_pred = clf.predict(X_val)

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
        test_params = {
            'price_limit': 45,
            'train_limit': 300,
            'fixed_model_params': {
                'ruleGenerator': 'Hybrid',
                'masterSolver': 'barrierCrossover',
                'numRulesToReturn': 100,
                'fairness_module': fairness_metric
            },
        }
        print(f'{model} {numSplits}-fold cross validation with complexity={param_grid["complexity"]}, '
              f'epsilon={param_grid["epsilon"]}, and fairness_metric={fairness_metric}')
        foldnum = 0
        for train_index, val_index in kf.split(X):
            foldnum += 1
            print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            train = pd.DataFrame(X_train)
            train.columns = ['X_' + str(i) for i in range(len(train.columns))]
            train['y'] = y_train

            val = pd.DataFrame(X_val)
            val.columns = ['X_' + str(i) for i in range(len(val.columns))]
            val['y'] = y_val

            train = train.astype(bool)
            val = val.astype(bool)

            # Set up reporting
            eps = param_grid['epsilon']
            res = TestResults(pname + ' ' + '(%d,%d)' % (eps, param_grid['complexity']) + '-' + str(foldnum), group=True)
            res.res['eps'] = eps
            res.res['C'] = param_grid['complexity']

            # Set hyperparameters
            test_params = test_params.copy()
            test_params['fixed_model_params']['epsilon'] = eps
            test_params['fixed_model_params']['ruleComplexity'] = param_grid['complexity']

            # Run CG
            saved_rules = None
            res, classif = runSingleTest(train.drop('y', axis=1).to_numpy(), train['y'].to_numpy(),
                                         train['X_0'].to_numpy(),
                                         val.drop('y', axis=1).to_numpy(), val['y'].to_numpy(),
                                         val['X_0'].to_numpy(),
                                         test_params,
                                         saved_rules, res, colGen=True, rule_filter=False)

            # save accuracy of fold
            accuracy.append(res.res['accuracy'])
            CG_EqOfOp.append(res.res['EqualOpportunity'])
            CG_HammingEqOdd.append(res.res['EqualizedOdds'])

    else:
        print("WARNING: please specify model type (either 'RUG' or 'FastSDTOpt' or 'binoct' or 'CG' or 'FairRUG')")
        return

    print(f'Mean Accuracy for these parameters: {np.mean(accuracy)}')
    print('---')

    param_grid_out = pd.DataFrame(param_grid, index=[0])
    param_grid_out['Accuracy'] = np.mean(accuracy)

    if model=='FairRUG':
        param_grid_out['Unfairness'] = np.mean(unfairness)
    if model=='CG' or model=='FairCG':
        if fairness_metric=='EqOfOp':
            param_grid_out['Unfairness'] = np.mean(CG_EqOfOp)
        elif fairness_metric=='HammingEqOdd' or fairness_metric=='unfair':
            param_grid_out['Unfairness'] = np.mean(CG_HammingEqOdd)

    return param_grid_out


def find_best_params(evaluation_parameters):
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
    if 'Unfairness' in evaluation_parameters.columns:
        evaluation_parameters = evaluation_parameters.drop('Accuracy', axis=1)
        idx = evaluation_parameters['Unfairness'].idxmin()
        best_params = evaluation_parameters.drop('Unfairness', axis=1).loc[idx]
    else:
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
    elif model == 'FastSDTOpt':
        scores['Nr of Rules'].append(FastSDTOpt_helpers.get_num_leaves(clf.tree_))
        scores['Avg. Rule Length'].append(FastSDTOpt_helpers.get_avg_rule_length(clf.tree_))
        scores['Avg. Nr. Rules per Sample'].append(1)
        scores['Avg. Rule Length per Sample'].append(FastSDTOpt_helpers.get_avg_rule_length_per_sample(clf.tree_, X_test))
        scores['Fit Time'].append(clf.runtime_)

    return scores


def write_results(pname, scores, path, binary, shape, best_params, param_grid, model, fairness_metric=None):
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
        if model == 'FairRUG':
            print(f'Fairness metric used: {fairness_metric}', file=f)
        print(f'Binary file used: {binary}', file=f)
        print(f'Dataset shape (train+test): {shape} \n', file=f)
        print(f'Parameters tried for grid search: \n {param_grid} \n', file=f)
        print(f'Best parameters: \n {best_params} \n' ,file=f)
        for method in scores.keys():
            txt = '{0}: \t {1:.4f} ({2:.4f})'.format(method,
                                                     np.mean(scores[method]), np.std(scores[method]))

            print(txt, file=f)

        print('<---\n', file=f)

    return

def CG_rules_per_sample(X, rules):
    K = []
    for rule in rules:
        K.append(np.all(X[:, rule.astype(np.bool_)], axis=1))
    preds_manual = np.sum(K, axis=0) > 0

    # for each sample, record the indeces of the rules used
    dict_rules_per_sample = {}
    for i, x in enumerate(X):
        dict_rules_per_sample[i] = []

    for rule_index, output in enumerate(K):
        output = list(output)
        indices = [i for i, x in enumerate(output) if x == True]
        for s in indices:
            dict_rules_per_sample[s].append(rule_index)

    # for each rule, record the length of this rule
    rule_lengths = {}
    for i, rule in enumerate(rules):
        rule_lengths[i] = sum(rule)

    # combine both:
    # for each sample, record the lengths of the rules used for that sample
    rule_lengths_per_sample = {}
    for i, x in enumerate(X):
        rule_lengths_per_sample[i] = []
    for key, value in dict_rules_per_sample.items():
        for v in value:
            rule_lengths_per_sample[key].append(rule_lengths[v])

    nr_rules = []
    for key, value in dict_rules_per_sample.items():
        nr_rules.append(len(value))
    nr_rules = [x for x in nr_rules if x != 0]

    avg_rule_length_sample = []
    for key, value in rule_lengths_per_sample.items():
        avg_rule_length_sample.append(np.mean(value))

    return np.nanmean(nr_rules), np.nanmean(avg_rule_length_sample)


def run(problem, pgrid, save_path = None,
        randomState = 0, testSize=0.3, numSplits=5, binary = True, write=True,
        model = 'RUG', target = 'y', data_path='./prepped_data/',save_splits=True,
        fairness_metric=None):

    if save_path is None:
        save_path = f'./results_w_{model}_manual/'


    pname = problem.__name__.upper()
    print(f'---------{pname}---------')
    # get data using prep_data()
    X_train, X_test, y_train, y_test = prep_data(problem, binary=binary,
                                                 randomState=randomState, testSize=testSize, target=target,
                                                 data_path=data_path, save_splits=save_splits)

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
                            model=model, data_path=data_path, fairness_metric=fairness_metric)
        # save accuracy result of cv in dataframe
        # evaluation_parameters = evaluation_parameters.append(param_grid_out, ignore_index=True)
        evaluation_parameters = pd.concat([evaluation_parameters, param_grid_out], ignore_index=True)

    # find best parameters based on average accuracy
    best_params = find_best_params(evaluation_parameters)
    print(f'Best parameters: {best_params}')
    # make sure that parameters have correct data type
    for k in best_params:
        if isinstance(best_params[k], np.integer):
            best_params[k] = int(best_params[k])

    # test model on hold-out test set (X_test)
    if model == 'RUG':
        clf_final = RUGClassifier(random_state=randomState, threshold=0.05, rule_length_cost=True)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train)
        y_pred = clf_final.predict(X_test)
        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)
    elif model == 'FastSDTOpt':
        clf_final = DL85Classifier(time_limit=300, desc=True)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train)
        y_pred = clf_final.predict(X_test)
        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)
    elif model == 'binoct':
        filename = f'{data_path}{pname}_train_complete.csv'
        startTime = time.time()
        lcb.main(["-f", filename, "-d", best_params['max_depth'], "-t", 300, "-p", 100])
        endTime = time.time()
        predictTime = endTime - startTime

        import predict_file
        y_pred, path_lengths = predict_file.main([f'./prepped_data/{pname}_test.csv'])
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
        # create folder to save files generated by CG
        path = './results_w_CG_manual/res/'
        if not os.path.exists(path):
            os.makedirs(path)

        test_params = {
            'price_limit': 45,
            'train_limit': 300,
            'fixed_model_params': {
                'ruleGenerator': 'Hybrid',
                'masterSolver': 'barrierCrossover',
                'numRulesToReturn': 100,
                'fairness_module': 'unfair'
            },
        }
        train = pd.DataFrame(X_train)
        train.columns = ['X_' + str(i) for i in range(len(train.columns))]
        train['y'] = y_train

        test = pd.DataFrame(X_test)
        test.columns = ['X_' + str(i) for i in range(len(test.columns))]
        test['y'] = y_test

        train = train.astype(bool)
        test = test.astype(bool)

        # Set up reporting
        eps = 1
        res = TestResults(pname + ' ' + '(%d,%d)' % (eps, best_params['complexity']))
        res.res['eps'] = eps
        res.res['C'] = best_params['complexity']

        # Set hyperparameters
        test_params = test_params.copy()
        test_params['fixed_model_params']['epsilon'] = eps
        test_params['fixed_model_params']['ruleComplexity'] = best_params['complexity']

        print('---TRAIN FINAL MODEL---')
        # Run CG
        saved_rules = None
        res, classif = runSingleTest(train.drop('y', axis=1).to_numpy(), train['y'].to_numpy(),
                                     train['X_0'].to_numpy(),
                                     test.drop('y', axis=1).to_numpy(), test['y'].to_numpy(),
                                     test['X_0'].to_numpy(),
                                     test_params,
                                     saved_rules, res, colGen=True, rule_filter=False)
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
    elif model == 'FairRUG':
        # Obtain classes and groups
        groups = pd.unique(X_train[:, 0])
        groups.sort()
        classes = pd.unique(y_train)
        classes.sort()

        # For each pair of groups, create sets P (list of vectors/np.array)
        constraintSetPairs_train, pairs = FC.create_setsPI(X_train, y_train, groups, metric=fairness_metric)
        constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric=fairness_metric)

        clf_final = RUGClassifier(random_state=randomState, threshold=0.05, fair_metric=fairness_metric)
        clf_final.set_params(**best_params)
        clf_final.fit(X_train, y_train, groups=constraintSetPairs_train)
        y_pred = clf_final.predict(X_test)
        # RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
        scores = get_results(y_test, y_pred, clf_final, X_test=X_test, model=model)

        if len(classes)==2 and fairness_metric=='dmc':
            RUG_EqualizedOdds = FC.binary_EqOdds(y_test, y_pred, constraintSetPairs_test, classes, pairs)
            RUG_EqualOpportunity = FC.binary_EqOpp(y_test, y_pred, constraintSetPairs_test, classes, pairs)
            scores['Equalized Odds'] = [1-RUG_EqualizedOdds]
            scores['Equal Opportunity'] = [1-RUG_EqualOpportunity]
        else:  # this includes odm, should be run separately from dmc
            RUG_unfairness = FC.fairnessEvaluation(y_test, y_pred, constraintSetPairs_test, classes, pairs)
            if len(classes) > 2:
                scores['Fairness'] = [1 - RUG_unfairness]
            else:
                scores['Fairness ODM'] = [1 - RUG_unfairness]

    elif model == 'FairCG':
        # create folder to save files generated by CG
        path = './results_w_FairCG_manual/res/'
        if not os.path.exists(path):
            os.makedirs(path)

        test_params = {
            'price_limit': 45,
            'train_limit': 300,
            'fixed_model_params': {
                'ruleGenerator': 'Hybrid',
                'masterSolver': 'barrierCrossover',
                'numRulesToReturn': 100,
                'fairness_module': fairness_metric
            },
        }
        train = pd.DataFrame(X_train)
        train.columns = ['X_' + str(i) for i in range(len(train.columns))]
        train['y'] = y_train

        test = pd.DataFrame(X_test)
        test.columns = ['X_' + str(i) for i in range(len(test.columns))]
        test['y'] = y_test

        train = train.astype(bool)
        test = test.astype(bool)

        # Set up reporting
        eps = best_params['epsilon']
        res = TestResults(pname + ' ' + '(%d,%d)' % (eps, best_params['complexity']), group=True)
        res.res['eps'] = eps
        res.res['C'] = best_params['complexity']

        # Set hyperparameters
        test_params = test_params.copy()
        test_params['fixed_model_params']['epsilon'] = eps
        test_params['fixed_model_params']['ruleComplexity'] = best_params['complexity']

        print('---TRAIN FINAL MODEL---')
        # Run CG
        saved_rules = None
        res, classif = runSingleTest(train.drop('y', axis=1).to_numpy(), train['y'].to_numpy(),
                                     train['X_0'].to_numpy(),
                                     test.drop('y', axis=1).to_numpy(), test['y'].to_numpy(),
                                     test['X_0'].to_numpy(),
                                     test_params,
                                     saved_rules, res, colGen=True, rule_filter=False)
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
        scores['Equalized Odds'] = [1-value for value in res.res['EqualizedOdds']]
        scores['EqualOpportunity'] = [1-value for value in res.res['EqualOpportunity']]
        scores['ODM'] = [1-value for value in res.res['ODM']]

    else:
        return

    if write:
        shape = (len(X_train)+len(X_test), X_train.shape[1]+1)
        write_results(pname, scores, path = save_path, binary = binary,
                      shape = shape, best_params=best_params,
                      param_grid = pgrid, model=model,
                      fairness_metric=fairness_metric)

    return

