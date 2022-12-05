"""
@author: sibirbil
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ruxg import RUXClassifier, RUGClassifier
import Datasets as DS

randomState = 21
maxDepth = 3
penpar = 1.0
solver = 'gurobi' # or 'glpk'

# CLASSIFICATION
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin,
            DS.seeds, DS.wine, DS.glass, DS.ecoli, DS.sensorless]

problems = [DS.banknote]

print('Random State, Maximum Depth: %d, %d\n' % (randomState, maxDepth))

for problem in problems: 
    pname = problem.__name__.upper()
    print(pname)
    
    df = np.array(problem('./datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=0.3)

    RF = RandomForestClassifier(max_depth=maxDepth, random_state=randomState)
    RF_pred = RF.fit(X_train, y_train).predict(X_test)
          
    print('Start RUXRF')
    RUXRF = RUXClassifier(trained_ensemble=RF,
                          pen_par=penpar,
                          rule_length_cost=True,
                          solver=solver,
                          random_state=randomState)
    RUXRF_pred = RUXRF.fit(X_train, y_train).predict(X_test)
  
    print('Start ADA')
    ADA = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=maxDepth),
                              algorithm='SAMME',
                              random_state=randomState)
    ADA_pred = ADA.fit(X_train, y_train).predict(X_test)

    print('Start RUXADA')    
    RUXADA = RUXClassifier(trained_ensemble=ADA,
                           pen_par=penpar,
                           rule_length_cost=True,
                           solver=solver,                            
                           random_state=randomState)
    RUXADA_pred = RUXADA.fit(X_train, y_train).predict(X_test)

    print('Start GB')
    GB = GradientBoostingClassifier(max_depth=maxDepth, random_state=randomState)
    GB_pred = GB.fit(X_train, y_train).predict(X_test)
    
    print('Start RUXGB')    
    RUXGB = RUXClassifier(trained_ensemble=GB,
                          pen_par=penpar,
                          rule_length_cost=True,
                          solver=solver,
                          random_state=randomState)
    RUXGB_pred = RUXGB.fit(X_train, y_train).predict(X_test)

    print('Start DT')
    DT = DecisionTreeClassifier(random_state=randomState)
    DT_pred = DT.fit(X_train, y_train).predict(X_test)

    print('Start RUG')
    RUG = RUGClassifier(max_depth=maxDepth,
                        pen_par=penpar,
                        rule_length_cost=True,
                        solver=solver,
                        random_state=randomState)
    RUG_pred = RUG.fit(X_train, y_train).predict(X_test)

    print('\n\n#### RESULTS #### \n')
    print('Accuracy of RF: ', accuracy_score(RF_pred, y_test)) 
    print('Number of rules in RF: ', RUXRF.get_init_num_of_rules())
    print('Accuracy of ADA: ', accuracy_score(ADA_pred, y_test))
    print('Number of rule in ADA: ', RUXADA.get_init_num_of_rules())
    print('Accuracy of GB: ', accuracy_score(GB_pred, y_test))
    print('Number of rules in GB: ', RUXGB.get_init_num_of_rules())
    print('Accuracy of DT:', accuracy_score(DT_pred, y_test))
    print('Number of rules in DT: ', DT.get_n_leaves())    
    print()
    
    print('Accuracy of RUX(RF): ', accuracy_score(RUXRF_pred, y_test))
    print('Number of rules in RUX(RF): ', RUXRF.get_num_of_rules())
    print('Average number of rules per sample in RUX(RF): ', RUXRF.get_avg_num_rules_per_sample())
    print('Training time for RUX(RF): ', RUXRF.get_fit_time())
    print('Prediction time for RUX(RF): ', RUXRF.get_predict_time())       
    print()
    print('Accuracy of RUX(ADA): ', accuracy_score(RUXADA_pred, y_test))    
    print('Number of rules in RUX(ADA): ', RUXADA.get_num_of_rules())
    print('Average number of rules per sample in RUX(ADA): ', RUXADA.get_avg_num_rules_per_sample())
    print('Training time for RUX(ADA): ', RUXADA.get_fit_time())
    print('Prediction time for RUX(ADA): ', RUXADA.get_predict_time())  
    print()
    print('Accuracy of RUX(GB): ', accuracy_score(RUXGB_pred, y_test))
    print('Number of rules in RUX(GB): ', RUXGB.get_num_of_rules())
    print('Average number of rules per sample in RUX(GB): ', RUXGB.get_avg_num_rules_per_sample())
    print('Training time for RUX(GB): ', RUXGB.get_fit_time())
    print('Prediction time for RUX(GB): ', RUXGB.get_predict_time())  
    print()
    print('Accuracy of RUG: ', accuracy_score(RUG_pred, y_test))
    print('Number of rules in RUG: ', RUG.get_num_of_rules())
    print('Average number of rules per sample in RUG: ', RUG.get_avg_num_rules_per_sample())
    print('Training time for RUG: ', RUG.get_fit_time())
    print('Prediction time for RUG: ', RUG.get_predict_time())
    
    print()
    

