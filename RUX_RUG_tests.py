#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21 April 2021

@author: sibirbil
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from rulediscovery import RUXClassifier, RUGClassifier
import Datasets as DS

randomState = 9
maxDepth = 2
rhsEps = 0.01

# NOTE: Choose Gurobi for large problems like skinnoskin
solver = 'glpk' # or 'gurobi'
    
problems = [DS.banknote, DS.seeds, DS.glass, DS.ecoli] 

for problem in problems: 
    pname = problem.__name__.upper()
    print(pname)
    
    df = np.array(problem('datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=0.3)
    
    RF = RandomForestClassifier(max_depth=maxDepth, random_state=randomState)
    RF_fit = RF.fit(X_train, y_train)
    RF_pred = RF_fit.predict(X_test)
          

    RUXRF = RUXClassifier(rf=RF_fit, eps=rhsEps,
                          rule_length_cost=True,
                          false_negative_cost=False, 
                          solver='glpk',
                          random_state=randomState)
    RUXRF_fit = RUXRF.fit(X_train, y_train)
    RUXRF_pred = RUXRF.predict(X_test)
  

    ADA = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=maxDepth),
                              algorithm='SAMME',
                              random_state=randomState)
    ADA_fit = ADA.fit(X_train, y_train)
    ADA_pred = ADA_fit.predict(X_test)

    
    RUXADA = RUXClassifier(ada=ADA_fit, eps=rhsEps, 
                            use_ada_weights=True,
                            solver=solver,                            
                            random_state=randomState)
    RUXADA_fit = RUXADA.fit(X_train, y_train)
    RUXADA_pred = RUXADA.predict(X_test)


    RUG = RUGClassifier(eps=rhsEps,
                        max_depth=maxDepth,
                        rule_length_cost=True,
                        false_negative_cost=False,
                        solver=solver,
                        random_state=randomState)
    RUG_fit = RUG.fit(X_train, y_train)
    RUG_pred = RUG.predict(X_test)


    print('\n\n#### RESULTS #### \n')
    print('Accuracy of RF: ', accuracy_score(RF_pred, y_test)) 
    
    print('Accuracy of RUX(RF): ', accuracy_score(RUXRF_pred, y_test))
    print('Total number of RF rules: ', RUXRF.getInitNumOfRules())
    print('Total number of rules in RUX(RF): ', RUXRF.getNumOfRules())
    print('Total number of missed samples in RUX(RF): ', RUXRF.getNumOfMissed())
    print('Training time for RUX(RF)', RUXRF.getFitTime())
    print('Prediction time for RUX(RF)', RUXRF.getPredictTime())       

    print('Accuracy of ADA: ', accuracy_score(ADA_pred, y_test))

    print('Accuracy of RUX(ADA): ', accuracy_score(RUXADA_pred, y_test))
    print('Total number of ADA rules: ', RUXADA.getInitNumOfRules())
    print('Total number of rules in RUX(ADA): ', RUXADA.getNumOfRules())
    print('Total number of missed samples in RUX(ADA): ', RUXADA.getNumOfMissed())
    print('Training time for RUX(ADA)', RUXADA.getFitTime())
    print('Prediction time for RUX(ADA)', RUXADA.getPredictTime())  

    print('Accuracy of RUG: ', accuracy_score(RUG_pred, y_test))
    print('Total number of rules in RUG: ', RUG.getNumOfRules())
    print('Total number of missed samples in RUG: ', RUG.getNumOfMissed())
    print('Training time for RUG', RUG.getFitTime())
    print('Prediction time for RUG', RUG.getPredictTime())  

    print()
