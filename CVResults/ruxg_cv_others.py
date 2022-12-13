"""
@author: sibirbil
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import Datasets as DS

numOutCV: int = 10
numInCV: int = 5

randomState = 16252329

path = './results_w_others/'
fname: str = path + 'others_cv_'

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
    RFestimator = RandomForestClassifier(random_state=randomState)
    ADAestimator = AdaBoostClassifier(random_state=randomState)
    GBestimator = GradientBoostingClassifier(random_state=randomState)
    KNNestimator = KNeighborsClassifier()
    DTestimator = DecisionTreeClassifier(random_state=randomState)
    SVCestimator = SVC()    
    MLPestimator = MLPClassifier(learning_rate='adaptive', max_iter=10000, random_state=randomState)
    LRestimator = LogisticRegression()
    
    # Setting up the parameter grids
    RF_pgrid = {'n_estimators': [100, 200]}
    ADA_pgrid = {'n_estimators': [50, 100]}
    GB_pgrid = {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [100, 200]}
    KNN_pgrid = {'n_neighbors': [5, 10, 20]}
    DT_pgrid = {'min_samples_split': [2, 4]}
    SVC_pgrid = {'C': [0.1, 1.0, 10.0]}
    MLP_pgrid = {'alpha': [0.0001, 0.001, 0.01]}
    LR_pgrid = {'C': [0.1, 1.0, 10.0]}
    
    scores = {'RF': [], 'ADA': [], 'GB': [],
              'KNN': [], 'DT': [], 'SVC':[], 'MLP': [], 
              'LR': []}
    
    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=randomState)

    foldnum = 0
    for train_index, test_index in skf.split(X, y):
        foldnum += 1
        print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=randomState)
        
        for pgrid, est, name in zip((RF_pgrid, ADA_pgrid, GB_pgrid,
                                     KNN_pgrid, DT_pgrid, SVC_pgrid, MLP_pgrid, 
                                     LR_pgrid),
                                    (RFestimator, ADAestimator, GBestimator,
                                     KNNestimator, DTestimator,  SVCestimator, MLPestimator, 
                                     LRestimator),
                                    ('RF', 'ADA', 'GB', 'KNN',
                                     'DT', 'SVC', 'MLP', 'LR')):

            gcv = GridSearchCV(estimator=est, param_grid=pgrid,
                                n_jobs=1, cv=inner_cv, verbose=0, refit=True)
            gcv_fit = gcv.fit(X_train, y_train)

            # Evaluate with the best estimator
            gcv_pred = gcv_fit.best_estimator_.predict(X_test)
            scores[name].append(accuracy_score(gcv_pred, y_test))

    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Method: Average & Std. Dev.\n', file=f)
                
        for method in scores.keys():
            txt = '{0}:  \t {1:.4f} \t {2:.4f}'.format(method, np.mean(scores[method]), np.std(scores[method]))
            print(txt, file=f)
            
        print('<---\n', file=f)

####################
# Solve all problems
for problem in problems:
    solve_problem(problem)
