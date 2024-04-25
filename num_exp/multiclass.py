import sys
sys.path.insert(1,'/Users/tabearober/OneDrive - UvA/Interpretable ML/03_RuleDiscovery/github/RuleDiscovery/num_exp')
import pandas as pd
import grid_search_helpers as gs_helpers
import Datasets as DS
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# multiclass
problems = [DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds]

numCV: int = 5
testSize = 0.2
randomState = 21

for problem in problems:
    pname = problem.__name__.upper()
    print(f'---------{pname}---------')

    results={}
    y_results = pd.DataFrame()
    X_train, X_test, y_train, y_test = gs_helpers.prep_data(problem, randomState=randomState, testSize=testSize,
                                                            target='y', model=None, use_binary=False)

    # --------------------------------
    print('Decision Tree')

    pgrid = {'max_depth': [3, 5, 7, 9, 11, 13, 15]}

    tc_estimator = DecisionTreeClassifier(criterion='gini', random_state=randomState)
    gcv = GridSearchCV(estimator=tc_estimator, param_grid=pgrid, n_jobs=1, cv=5, verbose=0, refit=True)
    gcv_fit = gcv.fit(X_train, y_train)

    tc = gcv_fit.best_estimator_
    params = tc.get_params()

    y_results['DT_pred'] = tc.predict(X_test)

    best_params = {key: params[key] for key in params.keys() & pgrid.keys()}
    results['Decision Tree'] = {'Parameters grid search':pgrid,
                                'Best parameters':best_params,
                                'Accuracy': round(accuracy_score(y_test, y_results['DT_pred']),4),
                                'weighted F1-score': round(f1_score(y_test, y_results['DT_pred'], average='weighted'),4),
                                'MCC-score': round(matthews_corrcoef(y_test, y_results['DT_pred']),4),
                                'Nr of rules': round(tc.get_n_leaves(),4),
                                'Average rule length': round(gs_helpers.average_depth(tc), 4),
                                'Average nr of rules per sample':1.00,
                                'Average rule length per sample':round(gs_helpers.average_path_length(tc, X_test),4),
                                'Fit time':round(gcv_fit.refit_time_,4)}

    # --------------------------------
    print('Random Forest')
    pgrid = {'max_depth': [3, 5, 7, 9, 11, 13, 15],
             'n_estimators': [100, 150, 200, 250, 300]}

    rf_estimator = RandomForestClassifier(criterion='gini', random_state=randomState)
    gcv = GridSearchCV(estimator=rf_estimator, param_grid=pgrid, n_jobs=1, cv=5, verbose=2, refit=True)
    gcv_fit = gcv.fit(X_train, y_train)

    rfc = gcv_fit.best_estimator_
    params = rfc.get_params()

    y_results['RF_pred'] = rfc.predict(X_test)

    n_leaves = 0
    for dtc in rfc.estimators_:
        n_leaves += dtc.get_n_leaves()

    best_params = {key: params[key] for key in params.keys() & pgrid.keys()}
    results['Random Forest'] = {'Parameters grid search': pgrid,
                                'Best parameters': best_params,
                                'Accuracy': round(accuracy_score(y_test, y_results['RF_pred']),4),
                                'weighted F1-score': round(f1_score(y_test, y_results['RF_pred'], average='weighted'),4),
                                'MCC-score': round(matthews_corrcoef(y_test, y_results['RF_pred']),4),
                                'Nr of rules': round(n_leaves,2),
                                'Average rule length': round(gs_helpers.avg_depth_ensemble(rfc), 4),
                                'Average nr of rules per sample': rfc.n_estimators,
                                'Average rule length per sample': round(gs_helpers.avg_path_length_ensemble(rfc, X_test),4),
                                'Fit time':round(gcv_fit.refit_time_,4)}

    #--------------------------------
    print('Adaboost')
    pgrid = {'n_estimators': [100, 150, 200, 250, 300]}

    ada_estimator = AdaBoostClassifier(random_state=randomState)
    gcv = GridSearchCV(estimator=ada_estimator, param_grid=pgrid, n_jobs=1, cv=5, verbose=2, refit=True)
    gcv_fit = gcv.fit(X_train, y_train)

    ada = gcv_fit.best_estimator_
    params = rfc.get_params()

    y_results['ADA_pred'] = ada.predict(X_test)

    best_params = {key: params[key] for key in params.keys() & pgrid.keys()}
    results['AdaBoost'] = {'Parameters grid search': pgrid,
                            'Best parameters': best_params,
                            'Accuracy': round(accuracy_score(y_test, y_results['ADA_pred']),4),
                            'weighted F1-score': round(f1_score(y_test, y_results['ADA_pred'], average='weighted'),4),
                            'MCC-score': round(matthews_corrcoef(y_test, y_results['ADA_pred']),4),
                            'Nr of rules': round(2*ada.n_estimators,2),
                           'Average rule length': 1.00,
                            'Average nr of rules per sample': ada.n_estimators,
                            'Average rule length per sample': 1.00,
                           'Fit time':round(gcv_fit.refit_time_,4)}

    # --------------------------------
    from lightgbm import LGBMClassifier
    print('LightGBM')
    pgrid = {'max_depth': [3, 5, 7, 9, 11, 13, 15],
             'n_estimators': [100, 150, 200, 250, 300]}

    lgbm_estimator = LGBMClassifier(random_state=randomState)
    gcv = GridSearchCV(estimator=lgbm_estimator, param_grid=pgrid, n_jobs=1, cv=5, verbose=1, refit=True)
    gcv_fit = gcv.fit(X_train, y_train)
    light_gbm = gcv_fit.best_estimator_
    params = light_gbm.get_params()

    y_results['LightGBM_pred'] = light_gbm.predict(X_test)

    n_leaves = sum(tree['num_leaves'] for tree in gcv_fit.best_estimator_._Booster.dump_model()["tree_info"])

    best_params = {key: params[key] for key in params.keys() & pgrid.keys()}
    results['LightGBM'] = {'Parameters grid search': pgrid,
                           'Best parameters': best_params,
                           'Accuracy': round(accuracy_score(y_test, y_results['LightGBM_pred']), 4),
                           'weighted F1-score': round(f1_score(y_test, y_results['LightGBM_pred'], average='weighted'),4),
                           'MCC-score': round(matthews_corrcoef(y_test, y_results['LightGBM_pred']), 4),
                           'Nr of rules': round(n_leaves, 2),
                           'Average rule length': round(gs_helpers.light_gbm_avg_rule_length(light_gbm), 4),
                           'Average nr of rules per sample': round(light_gbm._Booster.num_trees(), 4),
                           'Average rule length per sample': round(gs_helpers.light_gbm_avg_rule_length_per_sample(light_gbm, X_test), 4),
                           'Fit time': round(gcv_fit.refit_time_, 4)}

    # --------------------------------
    '''
    print('Gradient Boosting')
    pgrid = {'max_depth': [3, 5, 7, 9, 11, 13, 15],
             'n_estimators': [100, 150, 200, 250, 300]}

    gb_estimator = GradientBoostingClassifier(random_state=randomState)
    gcv = GridSearchCV(estimator=gb_estimator, param_grid=pgrid, n_jobs=1, cv=5, verbose=2, refit=True)
    gcv_fit = gcv.fit(X_train, y_train)

    gb = gcv_fit.best_estimator_
    params = gb.get_params()
    
    y_results['GB_pred'] = gb.predict(X_test)

    num_leaves = sum(tree.tree_.n_leaves for tree in gb.estimators_.reshape(-1))

    best_params = {key: params[key] for key in params.keys() & pgrid.keys()}
    results['Gradient Boosting'] = {'Parameters grid search': pgrid,
                                'Best parameters': best_params,
                                'Accuracy': round(accuracy_score(y_test, y_results['GB_pred']), 4),
                                'weighted F1-score': round(f1_score(y_test, y_results['GB_pred'], average='weighted'), 4),
                                'MCC-score': round(matthews_corrcoef(y_test, y_results['GB_pred']), 4),
                                'Nr of rules': round(num_leaves, 2),
                                'Average rule length': round(gs_helpers.avg_depth_ensemble(gb),4),
                                'Average nr of rules per sample': gb.n_estimators,
                                'Average rule length per sample': round(gs_helpers.avg_path_length_ensemble(gb, X_test),4),
                                    'Fit time':round(gcv_fit.refit_time_,4)}
    '''
    #--------------------------------

    # print(results)
    with open('./results_multiclass.txt', 'a') as f:
        print('---------------------------------------------------------------------', file=f)
        print(f'--->\n{pname}\n', file=f)

        for method in results.keys():
            print(f'--{method}--', file=f)
            for key in results[method].keys():
                txt = f'{key}: {str(results[method][key])}'
                print(txt, file=f)
            print('\n', file=f)

