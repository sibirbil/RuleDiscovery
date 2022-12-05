"""
@author: sibirbil
"""
import numpy as np
import pandas as pd
import fairconstraints as FC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from ruxg import RUXClassifier, RUGClassifier
import Datasets as DS
import time


randomState = 21    
numEstimators = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
maxDepth = 3
penpar = 1.0
solver = 'gurobi' # or 'glpk'
fairness_epsilon = 0.01
fairness_metric = 'odm'

fname: str = './FairnessResults/'+str(fairness_metric)+'/fairruxg_'

# Select datasets
# problems = [DS.compas, DS.adult_fairness, DS.default, DS.law, DS.attrition, DS.recruitment, DS.student, DS.nursery]
problems = [DS.nursery]

print('Random State, Maximum Depth: %d, %d\n' % (randomState, maxDepth))

for problem in problems: 
    pname = problem.__name__.upper()
    identifier=str(pname)+"_"+str(randomState)+"_"
    
    print(pname)
    print('Fairness_epsilon: ', fairness_epsilon)
    print('Fairness_metric: ', fairness_metric)
    
    df = np.array(problem('./datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]
    
    # Obtain classes and groups
    groups = pd.unique(X[:,0])
    groups.sort()  
    classes = pd.unique(y)
    classes.sort()
    print(classes, groups)

    # K-Fold
    kfold = KFold(10, shuffle=True, random_state=randomState)
    split_no = 1

    # For data output purposes
    GB_kfold = []
    RF_kfold = []
    ADA_kfold = []
    DT_kfold = []

    RUXGB_kfold = []
    RUXRF_kfold = []
    RUXADA_kfold = []
    RUG_kfold = []

    for train, test in kfold.split(df):
        print('K-Fold split ', str(split_no))
        X_train = df[train][:, 0:-1]
        y_train = df[train][:, -1]
        X_test = df[test][:, 0:-1]
        y_test = df[test][:, -1]

        # For data output purposes
        results_GB = []
        results_RF = []
        results_ADA = []
        results_DT = []

        results_RUXGB = []
        results_RUXRF = []
        results_RUXADA = []
        results_RUG = []
        
        # For each pair of groups, create sets P (list of vectors/np.array)
        constraintSetPairs_train, pairs = FC.create_setsPI(X_train, y_train, groups, metric=fairness_metric)
        constraintSetPairs_test, pairs = FC.create_setsPI(X_test, y_test, groups, metric=fairness_metric)

        # Start tests
        GB = GradientBoostingClassifier(n_estimators=numEstimators, max_depth=maxDepth, random_state=randomState)
        start = time.time()
        GB_pred = GB.fit(X_train, y_train).predict(X_test)
        GB_time = time.time()-start
        GB_unfairness = FC.fairnessEvaluation(y_test, GB_pred, constraintSetPairs_test, classes, pairs)

        RUXGB = RUXClassifier(trained_ensemble=GB,
                            pen_par=penpar,
                            rule_length_cost=True,
                            fair_eps=fairness_epsilon,
                            solver=solver,
                            fair_metric=fairness_metric,
                            random_state=randomState)
        RUXGB_pred = RUXGB.fit(X_train, y_train, groups=constraintSetPairs_train).predict(X_test)
        RUXGB_unfairness = FC.fairnessEvaluation(y_test, RUXGB_pred, constraintSetPairs_test, classes, pairs)


        RF = RandomForestClassifier(n_estimators=numEstimators, max_depth=maxDepth, random_state=randomState)
        start = time.time()
        RF_pred = RF.fit(X_train, y_train).predict(X_test)
        
        RF_time = time.time()-start
        RF_unfairness = FC.fairnessEvaluation(y_test, RF_pred, constraintSetPairs_test, classes, pairs)

        RUXRF = RUXClassifier(trained_ensemble=RF,
                            pen_par=penpar,
                            rule_length_cost=True,
                            fair_eps=fairness_epsilon,
                            solver=solver,
                            fair_metric=fairness_metric,
                            random_state=randomState)
        RUXRF_pred = RUXRF.fit(X_train, y_train, groups=constraintSetPairs_train).predict(X_test)
        RUXRF_unfairness = FC.fairnessEvaluation(y_test, RUXRF_pred, constraintSetPairs_test, classes, pairs)
    

        ADA = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=maxDepth),
                                random_state=randomState)
        start = time.time()                        
        ADA_pred = ADA.fit(X_train, y_train).predict(X_test)
        ADA_time = time.time()-start
        ADA_unfairness = FC.fairnessEvaluation(y_test, ADA_pred, constraintSetPairs_test, classes, pairs)


        RUXADA = RUXClassifier(trained_ensemble=ADA,
                            pen_par=penpar,
                            rule_length_cost=True,
                            fair_eps=fairness_epsilon,
                            solver=solver,
                            fair_metric=fairness_metric,            
                            random_state=randomState)
        RUXADA_pred = RUXADA.fit(X_train, y_train, groups=constraintSetPairs_train).predict(X_test)
        RUXADA_unfairness = FC.fairnessEvaluation(y_test, RUXADA_pred, constraintSetPairs_test, classes, pairs)

        RUG = RUGClassifier(max_depth=maxDepth,
                            pen_par=penpar,
                            rule_length_cost=True,
                            fair_eps=fairness_epsilon,
                            solver=solver,
                            fair_metric=fairness_metric,
                            random_state=randomState)
        RUG_pred = RUG.fit(X_train, y_train, groups=constraintSetPairs_train).predict(X_test)
        RUG_unfairness = FC.fairnessEvaluation(y_test, RUG_pred, constraintSetPairs_test, classes, pairs)

        DT = DecisionTreeClassifier(max_depth=maxDepth,
                                    random_state=randomState)
        start = time.time()
        DT_pred = DT.fit(X_train, y_train).predict(X_test)
        DT_time = time.time()-start
        DT_unfairness = FC.fairnessEvaluation(y_test, DT_pred, constraintSetPairs_test, classes, pairs)


        # GET RESULTS
        results_GB.append(round(accuracy_score(GB_pred,y_test),3)) 
        results_GB.append(round(RUXGB.get_init_num_of_rules(),3)) 
        results_GB.append(float("nan"))
        results_GB.append(float("nan"))
        results_GB.append(1-round(GB_unfairness,3))
        results_GB.append(round(GB_time,3))
        results_GB.append(round(GB_time,3))
        GB_kfold.append(results_GB)

        results_RF.append(round(accuracy_score(RF_pred,y_test),3)) 
        results_RF.append(round(RUXRF.get_init_num_of_rules(),3)) 
        results_RF.append(float("nan"))
        results_RF.append(float("nan"))
        results_RF.append(1-round(RF_unfairness,3))
        results_RF.append(round(RF_time,3))
        results_RF.append(round(RF_time,3))
        RF_kfold.append(results_RF)

        results_ADA.append(round(accuracy_score(ADA_pred,y_test),3)) 
        results_ADA.append(round(RUXADA.get_init_num_of_rules(),3)) 
        results_ADA.append(float("nan"))
        results_ADA.append(float("nan"))
        results_ADA.append(1-round(ADA_unfairness,3))
        results_ADA.append(round(ADA_time,3))
        results_ADA.append(round(ADA_time,3))
        ADA_kfold.append(results_ADA)

        results_DT.append(round(accuracy_score(DT_pred,y_test),3)) 
        results_DT.append(DT.get_n_leaves()) 
        results_DT.append(float("nan"))
        results_DT.append(float("nan"))
        results_DT.append(1-round(DT_unfairness,3))
        results_DT.append(round(DT_time,3))
        results_DT.append(round(DT_time,3))
        DT_kfold.append(results_DT)

        results_RUXGB.append(round(accuracy_score(RUXGB_pred,y_test),3)) 
        results_RUXGB.append(round(RUXGB.get_num_of_rules(),3)) 
        results_RUXGB.append(round(RUXGB.get_avg_rule_length(),3)) 
        results_RUXGB.append(round(RUXGB.get_avg_num_rules_per_sample(),3)) 
        results_RUXGB.append(1-round(RUXGB_unfairness,3)) 
        results_RUXGB.append(round(RUXGB.get_fit_time(),3)) 
        results_RUXGB.append(round(RUXGB.get_predict_time(),3)) 
        RUXGB_kfold.append(results_RUXGB)

        results_RUXRF.append(round(accuracy_score(RUXRF_pred,y_test),3))
        results_RUXRF.append(round(RUXRF.get_num_of_rules(),3)) 
        results_RUXRF.append(round(RUXRF.get_avg_rule_length(),3)) 
        results_RUXRF.append(round(RUXRF.get_avg_num_rules_per_sample(),3)) 
        results_RUXRF.append(1-round(RUXRF_unfairness,3)) 
        results_RUXRF.append(round(RUXRF.get_fit_time(),3)) 
        results_RUXRF.append(round(RUXRF.get_predict_time(),3)) 
        RUXRF_kfold.append(results_RUXRF)

        results_RUXADA.append(round(accuracy_score(RUXADA_pred,y_test),3)) 
        results_RUXADA.append(round(RUXADA.get_num_of_rules(),3)) 
        results_RUXADA.append(round(RUXADA.get_avg_rule_length(),3))
        results_RUXADA.append(round(RUXADA.get_avg_num_rules_per_sample(),3)) 
        results_RUXADA.append(1-round(RUXADA_unfairness,3)) 
        results_RUXADA.append(round(RUXADA.get_fit_time(),3))
        results_RUXADA.append(round(RUXADA.get_predict_time(),3))
        RUXADA_kfold.append(results_RUXADA)

        results_RUG.append(round(accuracy_score(RUG_pred,y_test),3)) 
        results_RUG.append(round(RUG.get_num_of_rules(),3)) 
        results_RUG.append(round(RUG.get_avg_rule_length(),3))
        results_RUG.append(round(RUG.get_avg_num_rules_per_sample(),3)) 
        results_RUG.append(1-round(RUG_unfairness,3)) 
        results_RUG.append(round(RUG.get_fit_time(),3))
        results_RUG.append(round(RUG.get_predict_time(),3))
        RUG_kfold.append(results_RUG)
        
        
        split_no=split_no+1

    print("----------------------------------------------------")
    print('###### K-fold averages #####')
    #----------- Existing methods
    row_RF = list(np.average(RF_kfold, axis=0))
    sd_acc = np.std(RF_kfold, axis=0)[0]
    sd_fairness = np.std(RF_kfold, axis=0)[4]
    row_RF.insert(0, 'RF')
    row_RF.insert(2, sd_acc)
    row_RF.insert(5, float("nan"))
    row_RF.insert(7, float("nan"))
    row_RF.insert(9, sd_fairness)

    row_ADA = list(np.average(ADA_kfold, axis=0))
    sd_acc = np.std(ADA_kfold, axis=0)[0]
    sd_fairness = np.std(ADA_kfold, axis=0)[4]
    row_ADA.insert(0, 'ADA')
    row_ADA.insert(2, sd_acc)
    row_ADA.insert(5, float("nan"))
    row_ADA.insert(7, float("nan"))
    row_ADA.insert(9, sd_fairness)

    row_GB = list(np.average(GB_kfold, axis=0))
    sd_acc = np.std(GB_kfold, axis=0)[0]
    sd_fairness = np.std(GB_kfold, axis=0)[4]
    row_GB.insert(0, 'GB')
    row_GB.insert(2, sd_acc)
    row_GB.insert(5, float("nan"))
    row_GB.insert(7, float("nan"))
    row_GB.insert(9, sd_fairness)

    row_DT = list(np.average(DT_kfold, axis=0))
    sd_acc = np.std(DT_kfold, axis=0)[0]
    sd_fairness = np.std(DT_kfold, axis=0)[4]
    row_DT.insert(0, 'DT')
    row_DT.insert(2, sd_acc)
    row_DT.insert(5, float("nan"))
    row_DT.insert(7, float("nan"))
    row_DT.insert(9, sd_fairness)

    #--------- RUXG
    row_RUXRF = list(np.average(RUXRF_kfold, axis=0))
    sd_acc = np.std(RUXRF_kfold, axis=0)[0]
    sd_fairness = np.std(RUXRF_kfold, axis=0)[4]
    sd_ARL = np.std(RUXRF_kfold, axis=0)[2]
    sd_ANoRpS = np.std(RUXRF_kfold, axis=0)[3]
    row_RUXRF.insert(0, 'RUX-RF')
    row_RUXRF.insert(2, sd_acc)
    row_RUXRF.insert(5, sd_ARL)
    row_RUXRF.insert(7, sd_ANoRpS)
    row_RUXRF.insert(9, sd_fairness)
    

    row_RUXADA = list(np.average(RUXADA_kfold, axis=0))
    sd_acc = np.std(RUXADA_kfold, axis=0)[0]
    sd_fairness = np.std(RUXADA_kfold, axis=0)[4]
    sd_ARL = np.std(RUXADA_kfold, axis=0)[2]
    sd_ANoRpS = np.std(RUXADA_kfold, axis=0)[3]
    row_RUXADA.insert(0, 'RUX-ADA')
    row_RUXADA.insert(2, sd_acc)
    row_RUXADA.insert(5, sd_ARL)
    row_RUXADA.insert(7, sd_ANoRpS)
    row_RUXADA.insert(9, sd_fairness)

    row_RUXGB = list(np.average(RUXGB_kfold, axis=0))
    sd_acc = np.std(RUXGB_kfold, axis=0)[0]
    sd_fairness = np.std(RUXGB_kfold, axis=0)[4]
    sd_ARL = np.std(RUXGB_kfold, axis=0)[2]
    sd_ANoRpS = np.std(RUXGB_kfold, axis=0)[3]
    row_RUXGB.insert(0, 'RUX-GB')
    row_RUXGB.insert(2, sd_acc)
    row_RUXGB.insert(5, sd_ARL)
    row_RUXGB.insert(7, sd_ANoRpS)
    row_RUXGB.insert(9, sd_fairness)

    row_RUG = list(np.average(RUG_kfold, axis=0))
    sd_acc = np.std(RUG_kfold, axis=0)[0]
    sd_fairness = np.std(RUG_kfold, axis=0)[4]
    sd_ARL = np.std(RUG_kfold, axis=0)[2]
    sd_ANoRpS = np.std(RUG_kfold, axis=0)[3]
    row_RUG.insert(0, 'RUG')
    row_RUG.insert(2, sd_acc)
    row_RUG.insert(5, sd_ARL)
    row_RUG.insert(7, sd_ANoRpS)
    row_RUG.insert(9, sd_fairness)

    # Report to DF
    df_full = pd.DataFrame(columns=['Method', 'Accuracy', 'SD (Acc)', 'NoR', \
        'ARL', 'SD (ARL)', 'ANoRpS', 'SD (ANoRpS)', 'Fairness', 'SD (Fairness)', \
            'Training time', 'Prediction time'])
    df_full.loc[len(df_full)] = row_RF
    df_full.loc[len(df_full)] = row_ADA
    df_full.loc[len(df_full)] = row_GB
    df_full.loc[len(df_full)] = row_DT

    df_full.loc[len(df_full)] = row_RUXRF
    df_full.loc[len(df_full)] = row_RUXADA
    df_full.loc[len(df_full)] = row_RUXGB
    df_full.loc[len(df_full)] = row_RUG
    print(df_full)

    fnamefull = fname + pname + '.txt'

    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname + ', sample size: ' + str(len(y)) + ', random state: ' + str(randomState), file=f)
        print('fairness metric: ' + str(fairness_metric) + ', epsilon: ' + str(fairness_epsilon), file=f)      
        print(df_full, file=f)
        print('<---\n', file=f)