import numpy as np
import pandas as pd
from binerizer import *
from DNFRuleModel import DNFRuleModel
from Classifier import Classifier
from sklearn.model_selection import train_test_split
import random
import time
import json
import copy

class TestResults(object):
    '''
    Object to contain and run the restricted model
    '''
    def __init__(self, testName, results_path = './results/', group = False):
        self.name = testName
        self.resultsPath = results_path
        self.evalGroups = group
        print(self.evalGroups)
     
    
        self.res = {}
        self.metrics = ['accuracy',
                        'accuracy_train',
                       'mip',
                       'mip_final',
                       'ip',
                       'complexity',
                       'eqOfOp',
                       'times',
                       'num_iter',
                       'num_cols_gen',
                       'hamming',
                       'hamming_pos',
                       'hamming_neg',
                       'hamming_tr',
                       'hamming_pos_tr',
                       'hamming_neg_tr',
                       'TPR1',
                       'TPR2',
                       'TNR1',
                       'TNR2',
                       'TPR-GAP',
                       'TNR-GAP',
                       'EqualOpportunity',
                       'ODM'
                       ]
        
        if self.evalGroups:
            self.metrics.append('accuracy_True')
            self.metrics.append('accuracy_False')
            self.metrics.append('eqOp_True')
            self.metrics.append('eqOp_False')
            self.metrics.append('eqOp_True_Train')
            self.metrics.append('eqOp_False_Train')
            self.metrics.append('accuracy_True_Train')
            self.metrics.append('accuracy_False_Train')
            self.metrics.append('hamming_True')
            self.metrics.append('hamming_False')
            self.metrics.append('hamming_tr_True')
            self.metrics.append('hamming_tr_False')
            self.metrics.append('hamming_pos_True')
            self.metrics.append('hamming_pos_False')
            self.metrics.append('hamming_pos_tr_True')
            self.metrics.append('hamming_pos_tr_False')
            self.metrics.append('hamming_neg_True')
            self.metrics.append('hamming_neg_False')
            self.metrics.append('hamming_neg_tr_True')
            self.metrics.append('hamming_neg_tr_False')


        for metric in self.metrics:
            self.res[metric] = []
        
    def update(self, classifier, time, test_data, groups = None):
        self.computeAccuracyMetrics(classifier, test_data, groups)
        self.computeHammingMetrics(classifier, test_data, groups)
        self.computeFairnessMetrics(classifier, test_data, groups)
        

        self.res['times'].append(time)
        self.res['complexity'].append(np.sum(classifier.fitRuleSet)+ len(classifier.fitRuleSet))
        self.res['num_iter'].append(classifier.numIter)
        
        if classifier.ruleMod.rules is None:
            self.res['num_cols_gen'].append(0)
        else:
            self.res['num_cols_gen'].append(len(classifier.ruleMod.rules))
        self.res['mip'].append(classifier.mip_results)
        self.res['mip_final'].append(classifier.final_mip)
        self.res['ip'].append(classifier.final_ip)
        
        # for metric in self.metrics:
        #     if metric == 'mip':
        #         continue
        #     elif metric in ['TPR-GAP','TNR-GAP']:
        #         # self.res['EqualizedOdds'] = max(self.res['TPR-GAP'], self.res['TNR-GAP']) #original
        #         # print('FNR-gap = ', self.res['TPR-GAP'])
        #         # print('FPR-gap = ', self.res['TNR-GAP'])
        #         #self.res['EqualizedOdds'] = max(self.res['TPR-GAP'], self.res['TNR-GAP']) #original
        #         self.res['EqualizedOdds'] = sum(self.res['TPR-GAP'],self.res['TNR-GAP']) #Adia: FNR=1-TPR and FPR=1-TNR. So no adjustment to FNR and FPR needed for the gap
        #         self.res['EqualOpportunity'] = self.res['TPR-GAP']
        #     else:
        #         self.res['mean_'+metric] = np.mean(self.res[metric])
        #         self.res['std_'+metric] = np.std(self.res[metric])
        
        self.printResult()
        try: self.write()
        except:
            return

        return
    
    def computeFairnessMetrics(self, classifier, test_data, groups):
        X_tr = classifier.ruleMod.X
        Y_tr = classifier.ruleMod.Y
        
        pos_preds = classifier.predict(test_data[0][groups]) #predictions group 1 on testdata
        neg_preds = classifier.predict(test_data[0][~groups]) #predictions group 2 on testdata

        TPR_Rate_1 = sum(pos_preds[test_data[1][groups]])/sum(test_data[1][groups])
        TPR_Rate_2 = sum(neg_preds[test_data[1][~groups]])/sum(test_data[1][~groups])

        TNR_Rate_1 = sum((pos_preds == 0) & (test_data[1][groups] == 0)) / sum(test_data[1][groups] == 0)
        TNR_Rate_2 = sum((neg_preds == 0) & (test_data[1][~groups] == 0)) / sum(test_data[1][~groups] == 0) 
               
        self.res['TPR1'].append(TPR_Rate_1) #TPR group 1?
        self.res['TPR2'].append(TPR_Rate_2) #TPR group 2
        self.res['TNR1'].append(TNR_Rate_1) #TNR group 1?
        self.res['TNR2'].append(TNR_Rate_2) #TNR group 2

        self.res['TPR-GAP'].append(abs(TPR_Rate_1-TPR_Rate_2))
        self.res['TNR-GAP'].append(abs(TNR_Rate_1-TNR_Rate_2))   

        #-------- EqualOpportunity
        self.res['EqualOpportunity'] = self.res['TPR-GAP']

        #-------- EqualizedOdds
        self.res['EqualizedOdds'] = sum(self.res['TPR-GAP'],self.res['TNR-GAP']) #Adia: FNR=1-TPR and FPR=1-TNR. So no adjustment to FNR and FPR needed for the gap

        #-------- ODM
        odm_count1 = 0
        odm_count2 = 0
        for i, is_group1 in enumerate(groups):
            if is_group1 == True:
                if test_data[1][i] != Y_tr[i]:
                    odm_count1 += 1
            else:
                if test_data[1][i] != Y_tr[i]:
                    odm_count2 += 1
        self.res['ODM'].append(abs((odm_count1/sum(groups)) - (odm_count2/sum(~groups))))



    def computeAccuracyMetrics(self, classifier, test_data, groups):
        X_tr = classifier.ruleMod.X
        Y_tr = classifier.ruleMod.Y

        self.res['accuracy'].append(sum(classifier.predict(test_data[0]) == test_data[1])/len(test_data[1]))
        self.res['accuracy_train'].append(sum(classifier.predict(X_tr) == Y_tr)/len(Y_tr))

        # if self.evalGroups:
        #     pos_preds = classifier.predict(test_data[0][groups]) #predictions group 1 on testdata
        #     neg_preds = classifier.predict(test_data[0][~groups]) #predictions group 2 on testdata
            
        #     self.res['accuracy_True'].append(sum(pos_preds == test_data[1][groups])/len(test_data[1][groups]))
        #     self.res['accuracy_False'].append(sum(neg_preds ==  test_data[1][~groups])/len(test_data[1][~groups]))
            
        #     self.res['eqOp_True'].append(sum(pos_preds[test_data[1][groups]])/sum(test_data[1][groups])) #TPR group 1?
        #     self.res['eqOp_False'].append(sum(neg_preds[test_data[1][~groups]])/sum(test_data[1][~groups])) #FPR group 2?

        #     #Adia:------
        #     TPR_Rate_1 = sum(pos_preds[test_data[1][groups]])/sum(test_data[1][groups])
        #     TPR_Rate_2 = sum(neg_preds[test_data[1][~groups]])/sum(test_data[1][~groups])

        #     TNR_Rate_1 = sum((pos_preds == 0) & (test_data[1][groups] == 0)) / sum(test_data[1][groups] == 0)
        #     TNR_Rate_2 = sum((neg_preds == 0) & (test_data[1][~groups] == 0)) / sum(test_data[1][~groups] == 0) 
            
        #     # self.res['TPR1'].append(sum(pos_preds[test_data[1][groups]])/sum(test_data[1][groups])) #TPR group 1?
        #     # self.res['TPR2'].append(sum(neg_preds[test_data[1][~groups]])/sum(test_data[1][~groups])) #TPR group ?
        #     # self.res['TNR1'].append(sum((pos_preds == 0) & (test_data[1][groups] == 0)) / sum(test_data[1][groups] == 0)) #TPR group 1?
        #     # self.res['TNR2'].append(sum((neg_preds == 0) & (test_data[1][~groups] == 0)) / sum(test_data[1][~groups] == 0)) #TPR group 1?
            
        #     self.res['TPR1'].append(TPR_Rate_1) #TPR group 1?
        #     self.res['TPR2'].append(TPR_Rate_2) #TPR group 2
        #     self.res['TNR1'].append(TNR_Rate_1) #TNR group 1?
        #     self.res['TNR2'].append(TNR_Rate_2) #TNR group 2

        #     self.res['TPR-GAP'].append(abs(TPR_Rate_1-TPR_Rate_2))
        #     self.res['TNR-GAP'].append(abs(TNR_Rate_1-TNR_Rate_2))

        #     # #-------- ODM
        #     # odm_count1 = 0
        #     # odm_count2 = 0
        #     # for i, is_group1 in enumerate(groups):
        #     #     if is_group1 == True:
        #     #         if test_data[1][i] != Y_tr[i]:
        #     #             odm_count1 += 1
        #     #     else:
        #     #         if test_data[1][i] != Y_tr[i]:
        #     #             odm_count2 += 1
            
        #     # # print('odm_count1 = ', odm_count1)
        #     # # print('odm_count2 = ', odm_count2)
        #     # # print('ODM = ', (odm_count1/sum(groups)) - (odm_count2/sum(~groups)))
        #     # # quit()
        #     # self.res['ODM'].append(abs((odm_count1/sum(groups)) - (odm_count2/sum(~groups))))


                

        #     groups_tr = classifier.fairnessModule.group
        #     pos_preds_tr = classifier.predict(X_tr[groups_tr])
        #     neg_preds_tr = classifier.predict(X_tr[~groups_tr])
           
        #     self.res['accuracy_True_Train'].append(sum(pos_preds_tr == Y_tr[groups_tr])/len(Y_tr[groups_tr]))
        #     self.res['accuracy_False_Train'].append(sum(neg_preds_tr ==  Y_tr[~groups_tr])/len(Y_tr[~groups_tr]))
            
        #     self.res['eqOp_True_Train'].append(sum(pos_preds_tr[Y_tr[groups_tr]])/sum(Y_tr[groups_tr]))
        #     self.res['eqOp_False_Train'].append(sum(neg_preds_tr[Y_tr[~groups_tr]])/sum(Y_tr[~groups_tr]))

    def computeHammingMetrics(self, classifier, test_data, groups):
        X_tr = classifier.ruleMod.X
        Y_tr = classifier.ruleMod.Y
        
        testHam = self.computeHamming(classifier.predictHamming(test_data[0]), test_data[1])
        self.res['hamming'].append(testHam[0])
        self.res['hamming_pos'].append(testHam[1])
        self.res['hamming_neg'].append(testHam[2])

        trainHam = self.computeHamming(classifier.predictHamming(X_tr), Y_tr)
        self.res['hamming_tr'].append(trainHam[0])
        self.res['hamming_pos_tr'].append(trainHam[1])
        self.res['hamming_neg_tr'].append(trainHam[2])
        
        if self.evalGroups:
            ham_true = self.computeHamming(classifier.predictHamming(test_data[0][groups]),test_data[1][groups])
            ham_false = self.computeHamming(classifier.predictHamming(test_data[0][~groups]),test_data[1][~groups])
            
            self.res['hamming_True'].append(ham_true[0])
            self.res['hamming_pos_True'].append(ham_true[1])
            self.res['hamming_neg_True'].append(ham_true[2])
            
            self.res['hamming_False'].append(ham_false[0])
            self.res['hamming_pos_False'].append(ham_false[1])
            self.res['hamming_neg_False'].append(ham_false[2])
            
            groups_tr = classifier.fairnessModule.group
            ham_true = self.computeHamming(classifier.predictHamming(X_tr[groups_tr]),Y_tr[groups_tr])
            ham_false = self.computeHamming(classifier.predictHamming(X_tr[~groups_tr]),Y_tr[~groups_tr])
            
            self.res['hamming_tr_True'].append(ham_true[0])
            self.res['hamming_pos_tr_True'].append(ham_true[1])
            self.res['hamming_neg_tr_True'].append(ham_true[2])
            
            self.res['hamming_tr_False'].append(ham_false[0])
            self.res['hamming_pos_tr_False'].append(ham_false[1])
            self.res['hamming_neg_tr_False'].append(ham_false[2])


    def computeHamming(self,preds, y, norm = True):
        ham_true = (preds[y] == 0).sum()
        ham_false = preds[~y].sum()
        ham = ham_true + ham_false
        if norm:
            return float(ham/len(y)), float(ham_true/y.sum()), float(ham_false/(~y).sum())
        else:
            return float(ham), float(ham_true), float(ham_false)

    def printResult(self):
        print('Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f'%(self.res['accuracy'][-1], 
                                                                                                 self.res['times'][-1], 
                                                                                                 self.res['complexity'][-1], 
                                                                                                 self.res['num_cols_gen'][-1], 
                                                                                                 self.res['num_iter'][-1]))
    def write(self):
        print(self.res)
        r = json.dumps(self.res)
        
        with open(self.resultsPath+self.name+".txt", "w") as text_file:
            text_file.write(r)


def checkArgs(args):
    #Do Sanity Testing on inputs
    print(args)
    if 'hyper_paramaters' not in args:
        print('No hyper parameters input. Running model with defaults!')
    else:
        for hp in args['hyper_paramaters']:
            print('%d parameters provided for hyper-parameter %s'%(len(args['hyper_paramaters'][hp]),hp))
    

def extractArgs(args):
    new_args = {}
    new_args['fixed_model_params'] = args['fixed_model_params'] if 'fixed_model_params' in args else {}
    new_args['price_limit'] = args['price_limit'] if 'price_limit' in args else 999999999
    new_args['train_limit'] = args['train_limit'] if 'train_limit' in args else 999999999
    new_args['num_hp_splits'] = args['num_hp_splits'] if 'num_hp_splits' in args else 3
    new_args['name'] = args['name'] 
    new_args['hyper_paramaters'] = args['hyper_paramaters']
    
    return new_args

def extractGlobalArgs(args):
    new_args = {}
    new_args['num_splits'] = args['num_splits'] if 'num_splits' in args else 10
    return new_args

def readData(dataInfo):
    data = pd.read_csv(dataInfo['file'])
    data = binerizeData(data)
    data = data.sample(frac=1).reset_index(drop=True)

    cols = data.columns
    group_index = [i for i in range(len(cols)) if cols[i] == dataInfo['groupCol']][0]
    data_X = data.to_numpy()[:,0:(data.shape[1]-1)]
    data_Y = data.to_numpy()[:,data.shape[1]-1]
    group = data.to_numpy()[:,group_index]
    
    return data_X, data_Y, group

def getFold(X,Y,group, indices):
    X_train = np.delete(X, indices, axis=0)
    Y_train = np.delete(Y, indices)
    g_train = np.delete(group, indices)

    X_test = X[indices,:]
    Y_test = Y[indices]
    g_test = group[indices]
    
    return X_train, Y_train, g_train, X_test, Y_test, g_test

def updateRuleSet(rule_set, rules):
    if rule_set is None:
        rule_set = rules
    else:
        rule_set = np.unique(np.concatenate([rule_set,rules]), axis = 0)
    
    return rule_set

def runSingleTest(X_tr, Y_tr, g_tr, X_t, Y_t, g_t, test_params, rules, res, colGen = True, rule_filter = False):
    """
    X_tr: train data X
    Y_tr: train data y
    g_tr: train data grouping variable
    X_t: test data X
    Y_t: test data y
    g_t: test data grouping variable
    test_params: dictionary with price_limit, train_limit, and fixed_model_params
        - fixed_model_params is again a dictionary with ruleGenerator, masterSolver, numRulesToReturn, fairness_module
    rules: saved_rules (start None)
    res: instance of class TestResults
    colGen: default True
    rule_filter: default False
    """
    test_params['fixed_model_params']['group'] = g_tr

    classif = Classifier(X_tr, Y_tr, test_params['fixed_model_params'], ruleGenerator = 'Generic')

    start_time = time.perf_counter() 
    classif.fit(initial_rules = rules,
                verbose = False, 
                timeLimit = test_params['train_limit'], 
                timeLimitPricing = test_params['price_limit'],
                colGen = colGen,
                rule_filter = rule_filter)
    time_to_exec = time.perf_counter() - start_time
                        
    res.update(classif, time_to_exec, (X_t, Y_t), g_t)
    
    return res, classif

def runNestedCV(X, Y, group, test_params, foldId = -1, colGen = True, rule_filter = False):
    saved_rules = None
    break_points_hp = np.floor(np.arange(0,1+1/test_params['num_hp_splits'],
                                         1/test_params['num_hp_splits'])*X.shape[0]).astype(np.int)
    
    print('Hyper parameters for CV: ', test_params['hyper_paramaters'])
    for hp in test_params['hyper_paramaters']:
        hp_results = {}#init results object
        
        for hp_val in test_params['hyper_paramaters'][hp]:
            
            res = TestResults(test_params['name']+' '+'(%s,%d)'%(hp, hp_val)+'-'+str(foldId))
            test_params['fixed_model_params'][hp] = hp_val
            res.res['hp'] = hp
            res.res['hp_val'] = hp_val

            hp_rules = None

            for j in range(test_params['num_hp_splits']):
                print('Split %d'%j)
                #Get fold
                X_tr, Y_tr, g_tr, X_t, Y_t, g_t = getFold(X,Y, group, np.arange(break_points_hp[j], break_points_hp[j+1]))
                res, classif = runSingleTest(X_tr, Y_tr, g_tr, X_t, Y_t, g_t, test_params, 
                                             hp_rules, res, colGen = colGen, rule_filter = rule_filter)
                        
                #Save any rules generated to use in final column gen prediction process
                rules = classif.ruleMod.rules
                saved_rules = updateRuleSet(saved_rules,rules)
                hp_rules = updateRuleSet(hp_rules,rules)
                    
            #Store quick access for accuracy mean result
            hp_results[str(hp_val)] = res.res['mean_accuracy']
                
        #Get optimal HP val
        optimal_param_value = [float(x) for x in hp_results][np.argmax([hp_results[x] for x in hp_results])]
        test_params['fixed_model_params'][hp] = optimal_param_value
    
    return test_params, saved_rules

def run_test(tests_raw, globalArgs, dataInfo, save_rule_set = False, results_path = './results/', verbose = False):
    globalRules = None
    globArgs = extractGlobalArgs(globalArgs)
    
    #Set-up Data
    X,Y,group = readData(dataInfo)
                
    #Set-up tests
    tests = []
    for test in tests_raw:
        checkArgs(test)
        tests.append(extractArgs(test))
    
    #Create results
    res = []
    for i in range(len(tests)):
        res.append(TestResults(tests[i]['name']))
        
    #Prepare data indices
    break_points = np.floor(np.arange(0,1+1/globArgs['num_splits'],1/globArgs['num_splits'])*X.shape[0]).astype(np.int)

    for i in range(globArgs['num_splits']):
        print('****** Running split %d ******'%i)
        
        #Get data for this fold
        X_train, Y_train, g_train, X_test, Y_test, g_test = getFold(X,Y, group, np.arange(break_points[i], break_points[i+1]))
        
        #Run every test for this fold
        for test in range(len(tests)):
            print('**** Running Test %s ****'%tests[test]['name'])
            final_params, saved_rules = runNestedCV(X_train, Y_train, g_train, tests[test], foldId = i)
            
            globalRules = updateRuleSet(globalRules, saved_rules)
            
            if save_rule_set:
                r = json.dumps(globalRules.tolist())
                with open(results_path+tests[test]['name']+'_rules.txt', "w") as text_file:
                    text_file.write(r)

            print('Running final model with parameters: '+ str(final_params['fixed_model_params']))
            res[test], _ = runSingleTest(X_train, Y_train, g_train, X_test, Y_test, g_test, final_params, saved_rules, res[test])

    return res, globalRules

def run_fairness_curve(tests_raw, globalArgs, dataInfo, inpuptRules, results_path = './results/', verbose = False):
    globalRules = None
    globArgs = extractGlobalArgs(globalArgs)
    
    #Set-up Data
    X,Y,group = readData(dataInfo)
                
    #Set-up tests
    tests = []
    for test in tests_raw:
        checkArgs(test)
        tests.append(extractArgs(test))
    
    #Create results
    res = []
    for i in range(len(tests)):
        res.append(TestResults(tests[i]['name']))
        
    #Prepare data indices
    break_points = np.floor(np.arange(0,1+1/globArgs['num_splits'],1/globArgs['num_splits'])*X.shape[0]).astype(np.int)

    for i in range(globArgs['num_splits']):
        print('****** Running split %d ******'%i)
        
        #Get data for this fold
        X_train, Y_train, g_train, X_test, Y_test, g_test = getFold(X,Y, group, np.arange(break_points[i], break_points[i+1]))
        
        #Run every test for this fold
        for test in range(len(tests)):
            final_params, saved_rules = runNestedCV(X_train, Y_train, g_train, tests[test], foldId = i, colGen = False)

            print('Running test with parameters: '+ str(tests[test]['fixed_model_params']))
            res[test], _ = runSingleTest(X_train, Y_train, g_train, 
                                         X_test, Y_test, g_test, final_params, 
                                         inpuptRules, res[test], colGen = False)

    return res, globalRules

def run_fairTest(epsilons, tests_raw, globalArgs, dataInfo, save_rule_set = False, 
                 results_path = './results/', verbose = False, test_complex = None):
    globArgs = extractGlobalArgs(globalArgs)
    
    #Set-up Data
    X,Y,group = readData(dataInfo)
                
    #Set-up tests
    tests = []
    for test in tests_raw:
        checkArgs(test)
        tests.append(extractArgs(test))
    
    #Create results
    res = []
    test_eps_res = []
    if test_complex is not None:
        for i in range(len(tests)):
            res.append(TestResults(tests[i]['name']))
            eps_res = []
            for eps in epsilons:
                for c in test_complex:
                    eps_res.append(TestResults(tests[i]['name']+'_'+str(eps)+'_c_'+str(c), group = True))

            test_eps_res.append(eps_res)

    else:
        for i in range(len(tests)):
            res.append(TestResults(tests[i]['name']))
            eps_res = []
            for eps in epsilons:
                eps_res.append(TestResults(tests[i]['name']+'_'+str(eps), group = True))

            test_eps_res.append(eps_res)
                
    #Prepare data indices
    break_points = np.floor(np.arange(0,1+1/globArgs['num_splits'],1/globArgs['num_splits'])*X.shape[0]).astype(np.int)

    for i in range(globArgs['num_splits']):
        print('****** Running split %d ******'%i)
        
        #Get data for this fold
        X_train, Y_train, g_train, X_test, Y_test, g_test = getFold(X,Y, group, np.arange(break_points[i], break_points[i+1]))
        
        fold_rules = None
        
        #Run every test for this fold
        for test in range(len(tests)):
            print('**** Running Test %s ****'%tests[test]['name'])
            final_params, saved_rules = runNestedCV(X_train, Y_train, g_train, tests[test], foldId = i)
            
            fold_rules = updateRuleSet(fold_rules, saved_rules)

            print('Running epsilons with generated rules: '+ str(final_params['fixed_model_params']))
            if test_complex is not None:
                test_eps_res[test] = run_fairness_curveComplex(epsilons, test_complex, test_eps_res[test],
                                                     final_params,
                                                     (X_train, Y_train, g_train),
                                                     (X_test, Y_test, g_test), 
                                                     fold_rules,
                                                     fold_id = i)
            else:
                test_eps_res[test] = run_fairness_curve2(epsilons, test_eps_res[test],
                                                     final_params,
                                                     (X_train, Y_train, g_train),
                                                     (X_test, Y_test, g_test), 
                                                     fold_rules,
                                                     fold_id = i)

    return eps_res

def run_fairness_curve2(epsilons, eps_results, final_params_raw, train_data, test_data, inpuptRules, results_path = './results/', verbose = False, fold_id = -1):
    #Run every test for this fold
    final_params = copy.deepcopy(final_params_raw)
    
    del final_params['fixed_model_params']['epsilon']
    del final_params['hyper_paramaters']['epsilon']

    for i in range(len(epsilons)):
        final_params['fixed_model_params']['epsilon'] =  epsilons[i]
        print('FINAL PARAMS: ', final_params)
        final_params, saved_rules = runNestedCV(train_data[0], train_data[1], train_data[2], final_params, 
                                                foldId = fold_id, colGen = False, rule_filter = True)

        print('Running CURVE test with parameters: '+ str(final_params['fixed_model_params']))
        eps_results[i], _ = runSingleTest(train_data[0], train_data[1], train_data[2],
                                     test_data[0], test_data[1], test_data[2], final_params,inpuptRules, 
                                            eps_results[i], colGen = False, rule_filter = True)

    return eps_results

def run_fairness_curveComplex(epsilons, ruleC, eps_results, final_params_raw, train_data, test_data, 
                              inpuptRules, results_path = './results/', verbose = False, fold_id = -1):
    #Run every test for this fold
    final_params = copy.deepcopy(final_params_raw)
    
    del final_params['fixed_model_params']['epsilon']
    if 'epsilon' in final_params['hyper_paramaters']:
        del final_params['hyper_paramaters']['epsilon']
    del final_params['fixed_model_params']['ruleComplexity']
    if 'ruleComplexity' in final_params['hyper_paramaters']:
        del final_params['hyper_paramaters']['ruleComplexity']
    
    res_ctr = 0
    for i in range(len(epsilons)):
        for j in range(len(ruleC)):
            final_params['fixed_model_params']['epsilon'] =  epsilons[i]
            final_params['fixed_model_params']['ruleComplexity'] =  ruleC[j]

            print('Running CURVE test with parameters: '+ str(final_params['fixed_model_params']))
            eps_results[res_ctr], _ = runSingleTest(train_data[0], train_data[1], train_data[2],
                                         test_data[0], test_data[1], test_data[2], final_params,inpuptRules, 
                                                eps_results[res_ctr], colGen = False, rule_filter = True)
            res_ctr +=1

    return eps_results