"""
@author: sibirbil
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from ruxg import RUGClassifier
import Datasets as DS

numOutCV: int = 10
numInCV: int = 5

randomState = 16252329

path = './results_w_rug/'
fname: str = path + 'rug_cv_'

# CLASSIFICATION
# problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
#             DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
#             DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
#             DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin,
#             DS.seeds, DS.wine, DS.glass, DS.ecoli, DS.sensorless]
problems = [DS.banknote]

def solve_problem(problem):

    if not os.path.exists(path):
        os.makedirs(path)

    pname = problem.__name__.upper()

    df = np.array(problem('../datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]
    
    # Initializing Regressors
    RUGestimator = RUGClassifier(random_state=randomState)
    
    # Setting up the parameter grids
    RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
                 'max_depth': [3, 5, 10],
                 'max_RMP_calls': [5, 15, 30]}
    
    scores = {'Accuracy': [],
              'Nr of Rules' : [],
              'Avg. Rule Length':[],
              'Avg. Nr. Rules per Sample':[]}
    
    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=randomState)

    foldnum = 0
    for train_index, test_index in skf.split(X, y):
        foldnum += 1
        print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=randomState)

        gcv = GridSearchCV(estimator=RUGestimator, param_grid=RUG_pgrid,
                            n_jobs=1, cv=inner_cv, verbose=0, refit=True)
        gcv_fit = gcv.fit(X_train, y_train)

        # Evaluate with the best estimator
        gcv_pred = gcv_fit.best_estimator_.predict(X_test)
        scores['Accuracy'].append(accuracy_score(gcv_pred, y_test))
        scores['Nr of Rules'].append(gcv_fit.best_estimator_.get_num_of_rules())
        scores['Avg. Rule Length'].append(gcv_fit.best_estimator_.get_avg_rule_length())
        scores['Avg. Nr. Rules per Sample'].append(gcv_fit.best_estimator_.get_avg_num_rules_per_sample())


    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('RUG \n', file = f)

        for method in scores.keys():
            txt = '{0}: \t {1:.4f} ({2:.4f})'.format(method,
                                                        np.mean(scores[method]), np.std(scores[method]))

            print(txt, file=f)
            
        print('<---\n', file=f)

####################
# Solve all problems
for problem in problems:
    solve_problem(problem)
