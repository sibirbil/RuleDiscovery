"""
@author: sibirbil
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import Datasets as DS
import ruxg

numOutCV: int = 10
numInCV: int = 5

randomState = 16252329


path = './results_w_rux/'
fname: str = path + 'rux_cv_'

# CLASSIFICATION
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin,
            DS.seeds, DS.wine, DS.glass, DS.ecoli, DS.sensorless]
problems = [DS.banknote]


def solve_problem(problem):

    if not os.path.exists(path):
        os.makedirs(path)

    pname = problem.__name__.upper()

    df = np.array(problem('../datasets/'))

    X = df[:, 0:-1]
    y = df[:, -1]
    
    # Initializing Regressors
    RFestimator = RandomForestClassifier(random_state=randomState)
    ADAestimator = AdaBoostClassifier(random_state=randomState)
    GBestimator = GradientBoostingClassifier(random_state=randomState)
    
    # Setting up the parameter grids
    RF_pgrid = {'n_estimators': [100, 200]}
    ADA_pgrid = {'n_estimators': [50, 100]}
    GB_pgrid = {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [100, 200]}
    RUX_pgrid = {'pen_par': [0.1, 1.0, 10.0]}
    
    scores = {'RF': [], 'ADA': [], 'GB': [],
              'RUX-RF-0.1':[], 'RUX-RF-1.0':[], 'RUX-RF-10.0':[],
              'RUX-ADA-0.1':[], 'RUX-ADA-1.0':[], 'RUX-ADA-10.0':[],
              'RUX-GB-0.1':[], 'RUX-GB-1.0':[], 'RUX-GB-10.0':[]}
    rux_rules = {'RF-0.1': [], 'ADA-0.1': [], 'GB-0.1': [],
              'RUX-RF-0.1':[], 'RUX-ADA-0.1':[], 'RUX-GB-0.1':[],
                 'RF-1.0': [], 'ADA-1.0': [], 'GB-1.0': [],
                 'RUX-RF-1.0': [], 'RUX-ADA-1.0': [], 'RUX-GB-1.0': [],
                 'RF-10.0': [], 'ADA-10.0': [], 'GB-10.0': [],
                 'RUX-RF-10.0': [], 'RUX-ADA-10.0': [], 'RUX-GB-10.0': []}
    rule_length = {'RF': np.nan, 'ADA': np.nan, 'GB': np.nan,
                   'RUX-RF-0.1':[], 'RUX-ADA-0.1':[], 'RUX-GB-0.1':[],
                   'RUX-RF-1.0': [], 'RUX-ADA-1.0': [], 'RUX-GB-1.0': [],
                   'RUX-RF-10.0': [], 'RUX-ADA-10.0': [], 'RUX-GB-10.0': []}
    sample_rules = {'RF': np.nan, 'ADA': np.nan, 'GB': np.nan,
                    'RUX-RF-0.1':[], 'RUX-ADA-0.1':[], 'RUX-GB-0.1':[],
                   'RUX-RF-1.0': [], 'RUX-ADA-1.0': [], 'RUX-GB-1.0': [],
                   'RUX-RF-10.0': [], 'RUX-ADA-10.0': [], 'RUX-GB-10.0': []}
    
    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=randomState)

    foldnum = 0
    for train_index, test_index in skf.split(X, y):
        foldnum += 1
        print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=randomState)
        
        for pgrid, est, name in zip((RF_pgrid, ADA_pgrid, GB_pgrid),
                                    (RFestimator, ADAestimator, GBestimator),
                                    ('RF', 'ADA', 'GB')):

            gcv = GridSearchCV(estimator=est, param_grid=pgrid,
                                n_jobs=1, cv=inner_cv, verbose=0, refit=True)
            gcv_fit = gcv.fit(X_train, y_train)

            # Evaluate with the best estimator
            ensemble_model = gcv_fit.best_estimator_
            gcv_pred = ensemble_model.predict(X_test)

            ### RUX ###
            # use best estimator as trained ensemble for RUX
            # dictionary with three models, each with different pen_par
            rux_models = {}
            rux_models['0.1'] = ruxg.RUXClassifier(trained_ensemble=ensemble_model, pen_par=0.1,
                                           rule_length_cost=True, random_state=randomState)
            rux_models['1.0'] = ruxg.RUXClassifier(trained_ensemble=ensemble_model, pen_par=1.0,
                                           rule_length_cost=True, random_state=randomState)
            rux_models['10.0'] = ruxg.RUXClassifier(trained_ensemble=ensemble_model, pen_par=0.10,
                                           rule_length_cost=True, random_state=randomState)

            # fit models and predict
            gcv_pred_1 = rux_models['0.1'].fit(X_train, y_train).predict(X_test)
            gcv_pred_2 = rux_models['1.0'].fit(X_train, y_train).predict(X_test)
            gcv_pred_3 = rux_models['10.0'].fit(X_train, y_train).predict(X_test)

            ## SAVE RESULTS OF EACH FOLD
            # save accuracy scores in dictionary
            scores[name].append(accuracy_score(gcv_pred, y_test))
            scores['RUX-%s-0.1' % name].append(accuracy_score(gcv_pred_1, y_test))
            scores['RUX-%s-1.0' % name].append(accuracy_score(gcv_pred_2, y_test))
            scores['RUX-%s-10.0' % name].append(accuracy_score(gcv_pred_3, y_test))

            # nr of rules
            rux_rules['%s-0.1' % name].append(round(rux_models['0.1'].get_init_num_of_rules(), 3))
            rux_rules['RUX-%s-0.1' % name].append(round(rux_models['0.1'].get_num_of_rules(), 3))
            rux_rules['%s-1.0' % name].append(round(rux_models['1.0'].get_init_num_of_rules(), 3))
            rux_rules['RUX-%s-1.0' % name].append(round(rux_models['1.0'].get_num_of_rules(), 3))
            rux_rules['%s-10.0' % name].append(round(rux_models['10.0'].get_init_num_of_rules(), 3))
            rux_rules['RUX-%s-10.0' % name].append(round(rux_models['10.0'].get_num_of_rules(), 3))

            # average rule length for rux
            rule_length['RUX-%s-0.1' % name].append(round(rux_models['0.1'].get_avg_rule_length(),3))
            rule_length['RUX-%s-1.0' % name].append(round(rux_models['1.0'].get_avg_rule_length(), 3))
            rule_length['RUX-%s-10.0' % name].append(round(rux_models['10.0'].get_avg_rule_length(), 3))

            # average nr of rules per sample
            sample_rules['RUX-%s-0.1' % name].append(round(rux_models['0.1'].get_avg_num_rules_per_sample(), 3))
            sample_rules['RUX-%s-1.0' % name].append(round(rux_models['1.0'].get_avg_num_rules_per_sample(), 3))
            sample_rules['RUX-%s-10.0' % name].append(round(rux_models['10.0'].get_avg_num_rules_per_sample(), 3))

    # get mean scores for each RUX model -- necessary because we then select the model with the highest mean score
    rux_rf_mean_scores = {'RUX-RF-0.1':np.mean(scores['RUX-RF-0.1']),
                          'RUX-RF-1.0':np.mean(scores['RUX-RF-1.0']),
                          'RUX-RF-10.0':np.mean(scores['RUX-RF-10.0'])}
    rux_ada_mean_scores = {'RUX-ADA-0.1':np.mean(scores['RUX-ADA-0.1']),
                          'RUX-ADA-1.0':np.mean(scores['RUX-ADA-1.0']),
                          'RUX-ADA-10.0':np.mean(scores['RUX-ADA-10.0'])}
    rux_gb_mean_scores = {'RUX-GB-0.1':np.mean(scores['RUX-GB-0.1']),
                          'RUX-GB-1.0':np.mean(scores['RUX-GB-1.0']),
                          'RUX-GB-10.0':np.mean(scores['RUX-GB-10.0'])}

    # determine best model
    # then get the dictionary keys right, such that we can easily extract the mean scores later when writing the results to .txt file
    rux_rf_best = max(rux_rf_mean_scores, key= lambda x: rux_rf_mean_scores[x]) # highest mean score
    scores['RUX-RF'] = scores.pop(rux_rf_best)
    rux_rules['RUX-RF'] = rux_rules.pop(rux_rf_best)
    rux_rules['RF'] = rux_rules.pop('RF-%s' % rux_rf_best.split('-')[2])
    rule_length['RUX-RF'] = rule_length.pop(rux_rf_best)
    sample_rules['RUX-RF'] = sample_rules.pop(rux_rf_best)

    rux_ada_best = max(rux_ada_mean_scores, key= lambda x: rux_ada_mean_scores[x])
    scores['RUX-ADA'] = scores.pop(rux_ada_best)
    rux_rules['RUX-ADA'] = rux_rules.pop(rux_ada_best)
    rux_rules['ADA'] = rux_rules.pop('ADA-%s' % rux_ada_best.split('-')[2])
    rule_length['RUX-ADA'] = rule_length.pop(rux_ada_best)
    sample_rules['RUX-ADA'] = sample_rules.pop(rux_ada_best)

    rux_gb_best = max(rux_gb_mean_scores, key=lambda x: rux_gb_mean_scores[x])
    scores['RUX-GB'] = scores.pop(rux_gb_best)
    rux_rules['RUX-GB'] = rux_rules.pop(rux_gb_best)
    rux_rules['GB'] = rux_rules.pop('GB-%s' % rux_gb_best.split('-')[2])
    rule_length['RUX-GB'] = rule_length.pop(rux_gb_best)
    sample_rules['RUX-GB'] = sample_rules.pop(rux_gb_best)

    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Method: \t Accuracy \t Std. Dev. \t Nr of Rules \t Std. Dev. \t Avg. Rule Length \t Std. Dev. \t Avg # Rules Sample \t Std. Dev. \n', file=f)

        for method in ['RF', 'ADA', 'GB', 'RUX-RF', 'RUX-ADA', 'RUX-GB']:
            txt = '{0}: \t {1:.4f} \t {2:.4f} \t {3:.4f} \t {4:.4f} \t {5:.4f} \t {6:.4f} \t {7:.4f} \t {8:.4f}'.format(method,
                                                                            np.mean(scores[method]),
                                                                            np.std(scores[method]),
                                                                            np.mean(rux_rules[method]),
                                                                            np.std(rux_rules[method]),
                                                                            np.mean(rule_length[method]),
                                                                            np.std(rule_length[method]),
                                                                            np.mean(sample_rules[method]),
                                                                            np.std(sample_rules[method]))
            print(txt, file=f)
            
        print('<---\n', file=f)

    return scores, rux_rules, rux_rf_mean_scores

####################
# Solve all problems
for problem in problems:
    solve_problem(problem)
