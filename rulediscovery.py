#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21 April 2021

@author: sibirbil
"""
import copy
import time
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
from auxClasses import Clause, Rule

class RUXClassifier:
    
        def __init__(self, rf=None, 
                     ada=None, 
                     eps=1.0e-2,
                     threshold=1.0e-6,
                     use_ada_weights=False,
                     rule_length_cost=False, 
                     false_negative_cost=False,
                     negative_label=1.0, # For identifying false negatives
                     solver='gurobi',
                     random_state=2516):
            
            self.eps = eps
            self.threshold = threshold
            self.wscale = 1.0
            self.vscale = 1.0
            self.fittedDTs = {}
            self.solver = solver
            self.randomState = random_state
            self.initNumOfRules = 0
            self.rules = {}      
            self.ruleInfo = {}
            self.K = None # number of classes
            self.labelToInteger = {} # mapping classes to integers
            self.integerToLabel= {} # mapping integers to classes
            self.vecY = None        
            self.majorityClass = None # which class is the majority
            self.missedXvals = None        
            self.numOfMissed = None
            # The following three vectors keep the Abar and A matrices
            # Used for CSR sparse matrices
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            # The cost of each rule is stored
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.ruleLengthCost = rule_length_cost
            self.falseNegativeCost = false_negative_cost
            self.negativeLabel = negative_label
            self.useADAWeights = use_ada_weights
            self.estimatorWeights = [] # Used with AdaBoost
            # For time keeping        
            self.fitTime = 0
            self.predictTime = 0
            # Classifier type
            self.classifier = ''
            
            self._checkOptions(rf, ada)

        def _checkOptions(self, rf, ada):
            
            if (rf == None and ada == None):
                print('RF or ADA should be provided')
                print('Exiting...')
                return None
            
            if (rf != None and ada != None):
                print('Both RF and ADA are provided')
                print('Proceeding with RF')
                ada = None
                
            if (rf != None):
                self.classifier = 'RF'
                
                if (self.useADAWeights):
                    print('Estimator weights work only with ADA')
                    self.useADAWeights = False
                    
                if (np.sum([self.ruleLengthCost, self.falseNegativeCost]) > 1):
                    print('Works with only one type of cost')
                    print('Proceeding with rule length')
                    self.falseNegativeCost = False
                    self.ruleLengthCost = True

                for treeno, fitTree in enumerate(rf.estimators_):
                    self.initNumOfRules += fitTree.get_n_leaves()
                    self.fittedDTs[treeno] = fitTree
                
            if (ada != None):
                self.classifier = 'ADA'
                
                if (ada.get_params()['algorithm'] != 'SAMME' and self.useADAWeights):
                    print('Estimator weights only work with SAMME algorithm of ADA')
                    print('Proceeding without estimator weights')
                    self.useADAWeights = False
                
                if (np.sum([self.ruleLengthCost, 
                            self.falseNegativeCost, 
                            self.useADAWeights]) > 1):
                    print('Works with only one type of cost')
                    print('Proceeding with estimator weights')
                    self.falseNegativeCost = False
                    self.ruleLengthCost = False
                    self.useADAWeights = True

                if (self.useADAWeights):
                    self.estimatorWeights = (1.0/(ada.estimator_weights_+1.0e-4))
                    
                for treeno, fitTree in enumerate(ada.estimators_):
                    self.initNumOfRules += fitTree.get_n_leaves()
                    self.fittedDTs[treeno] = fitTree

        def _cleanup(self):
            
            self.fittedDTs = {}   
            self.rules = {}      
            self.ruleInfo = {}
            self.labelToInteger = {} 
            self.integerToLabel= {}
            self.missedXvals = None        
            self.numOfMissed = None
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.estimatorWeights = []            
                    
        def _getRule(self, fitTree, nodeid):
            
            if (fitTree.tree_.feature[0] == -2): # No rule case
                return Rule()
            left = fitTree.tree_.children_left
            right = fitTree.tree_.children_right
            threshold = fitTree.tree_.threshold
        
            def recurse(left, right, child, returnRule=None):
                if returnRule is None:
                    returnRule = Rule()                
                if child in left: # 'l'
                    parent = np.where(left == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    ub=threshold[parent])
                else: # 'r'               
                    parent = np.where(right == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    lb=threshold[parent])                
                returnRule.addClause(clause)
                if parent == 0:
                    return returnRule
                else:
                    return recurse(left, right, parent, returnRule)
        
            retRule = recurse(left, right, nodeid)
        
            return retRule
        
        
        def _getMatrix(self, X, y, fitTree, treeno):

            if (len(self.cols) == 0):
                col = 0
            else:
                col = max(self.cols) + 1 # Next column        
            y_rules = fitTree.apply(X) # Tells us which sample is in which leaf            
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                leafYvals = y[covers] # y values of the samples in the leaf
                uniqueLabels, counts = np.unique(leafYvals, return_counts=True)
                label = uniqueLabels[np.argmax(counts)] # majority class in the leaf
                labelVector = np.ones(self.K)*(-1/(self.K-1))
                labelVector[self.labelToInteger[label]] = 1
                fillAhat = np.dot(self.vecY[:, covers].T, labelVector)
                self.rows = np.hstack((self.rows, covers))
                self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32)*col))
                self.yvals = np.hstack((self.yvals, np.ones(len(covers), dtype=np.float)*fillAhat))
                if (self.falseNegativeCost):
                    cost = 1.0
                    if (label != self.negativeLabel and self.negativeLabel in leafYvals):
                        cost += np.exp(counts[int(self.negativeLabel)]/np.sum(counts))
                elif (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                elif (self.useADAWeights):
                    cost = self.estimatorWeights[treeno]
                else:
                    cost = 1.0
                self.costs = np.append(self.costs, cost)
                self.ruleInfo[col] = (treeno, leafno, label)
                col += 1            

        def _getMatrices(self, X, y):
            
            for treeno, fitTree in enumerate(self.fittedDTs.values()):                    
                self._getMatrix(X, y, fitTree, treeno)

        def _preprocess(self, X, y):
            
            classes, classCounts = np.unique(y, return_counts=True)
            self.majorityClass = classes[np.argmax(classCounts)]
            for i, c in enumerate(classes):
                self.labelToInteger[c] = i
                self.integerToLabel[i] = c
            self.K = len(classes)
            n = len(y)
            self.vscale = 1.0
            self.vecY = np.ones((self.K, n))*(-1/(self.K-1))
            for i, c in enumerate(y):
                self.vecY[self.labelToInteger[c], i] = 1        
            
        def _fillRules(self, weights):
            
            weights = weights/np.max(weights) # Scaled weights
            selectedColumns = np.where(weights > self.threshold)[0] # Selected columns
            weightOrder = np.argsort(-weights[selectedColumns]) # Ordered weights
            orderedColumns = selectedColumns[weightOrder] # Ordered indices
            
            for i, col in enumerate(orderedColumns):
                treeno, leafno, label = self.ruleInfo[col]
                fitTree = self.fittedDTs[treeno]
                if (fitTree.get_n_leaves()==1):
                    self.rules[i] = Rule(label=self.majorityClass,
                                         clauses=[],
                                         weight=weights[col]) # No rule
                else:
                    self.rules[i] = self._getRule(fitTree, leafno)
                    self.rules[i].label = label
                    self.rules[i].weight = weights[col]                
                    self.rules[i]._cleanRule()

        def _solvePrimal(self):
            
            if(self.solver == 'glpk'):
                return self._solvePrimalGLPK()
            elif (self.solver == 'gurobi'):
                return self._solvePrimalGurobi()
            else:
                print('This solver does not exist')
                
                     
        def _solvePrimalGLPK(self):
            
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)        
            
            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale
            # Variables
            vs = cp.Variable(n, nonneg=True)
            ws = cp.Variable(m, nonneg=True)
            # Primal Model
            primal = cp.Problem(cp.Minimize((np.ones(n)*self.vscale) @ vs + 
                                   self.costs @ ws),
                              [(((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0,
                               A @ ws >= self.eps])
            primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
            
            return ws.value
            
        def _solvePrimalGurobi(self):
    
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)        
            
            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale
            # Primal Model
            modprimal = gp.Model('RUG Primal')
            modprimal.setParam('OutputFlag', False)
            # variables
            vs = modprimal.addMVar(shape=int(n), name='vs')
            ws = modprimal.addMVar(shape=int(m), name='ws') 
            # objective
            modprimal.setObjective((np.ones(n)*self.vscale) @ vs + 
                                   self.costs @ ws, GRB.MINIMIZE)
            # constraints
            modprimal.addConstr((((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0, name='Ahat Constraints')
            modprimal.addConstr(A @ ws >= self.eps, name='A Constraints')
            modprimal.optimize()
            
            return ws.X
        
        def printRules(self, indices=[]):
            
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING! printRules() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                if (rule == 'NR'):
                    print('==> No Rule: Set Majority Class')
                else:
                    rule.printRule()
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
    
        def printWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: printWeights() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
                
        def getWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: getWeights() ###\n')
                print('Do not have that many rules')                
                return None 
            
            return [self.rules[indx].weight for indx in indices]    
                
        def predict(self, X, indices=[]):       
            
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: predict() ###\n')
                print('Do not have that many rules')
                return None  
    
            self.missedXvals = []        
            self.numOfMissed = 0
            
            startTime = time.time()
            # TODO: Can be done in parallel
            returnPrediction = []
            for x0 in X:
                sumClassWeights = np.zeros(self.K)
                for indx in indices:
                    rule = self.rules[indx]
                    if (rule != 'NR'):
                        if(rule.checkRule(x0)):
                            lab2int = self.labelToInteger[rule.label]
                            sumClassWeights[lab2int] += rule.weight
                
                if (np.sum(sumClassWeights) == 0):
                    # Unclassified test sample
                    self.numOfMissed += 1
                    self.missedXvals.append(x0)
                    # Assigned to a class with the initial DT
                    getClass = self.fittedDTs[0].predict(x0.reshape(1, -1))[0]
                    returnPrediction.append(getClass)
                else:
                    sel_label_indx = np.argmax(sumClassWeights)
                    int2lab = self.integerToLabel[sel_label_indx]
                    returnPrediction.append(int2lab)
    
            endTime = time.time()
            self.predictTime = endTime - startTime
            
            return returnPrediction
    
        def getAvgRuleLength(self):
            
            return np.mean([rule.length() for rule in self.rules.values()])
            
        def getNumOfRules(self):
            
            return len(self.rules)
    
        def getInitNumOfRules(self):
            
            return self.initNumOfRules
    
        def getNumOfMissed(self):
            
            return self.numOfMissed
    
        def getFitTime(self):
            
            return self.fitTime
    
        def getPredictTime(self):
            
            return self.predictTime
        
        def fit(self, X, y):
            
            if (len(self.cols) != 0):
                self._cleanup()
       
            startTime = time.time()
            
            self._preprocess(X, y)
            self._getMatrices(X, y)
            
            ws = self._solvePrimal()

            self._fillRules(ws)
            
            endTime = time.time()
            
            self.fitTime = endTime - startTime
            
class RUGClassifier:       
        def __init__(self,
                     eps=1.0e-2,
                     threshold=1.0e-6,
                     max_depth=2,
                     max_RMP_calls=30,
                     rule_length_cost=False,
                     false_negative_cost=False,
                     negative_label=1.0, # For identifying false negatives
                     solver='gurobi',
                     random_state=2516):
           
            self.eps = eps
            self.threshold = threshold
            self.wscale = 1.0
            self.vscale = 1.0
            self.fittedDTs = {}
            self.solver = solver
            self.randomState = random_state
            self.rules = {}
            self.ruleInfo = {}
            self.K = None # number of classes
            self.labelToInteger = {} # mapping classes to integers
            self.integerToLabel= {} # mapping integers to classes
            self.vecY = None
            self.majorityClass = None # which class is the majority
            self.missedXvals = None
            self.numOfMissed = None
            self.maxDepth = max_depth
            self.maxRMPcalls = max_RMP_calls
            # The following three vectors keep the Abar and A matrices
            # Used for CSR sparse matrices
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32)
            # The cost of each rule is stored
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.ruleLengthCost = rule_length_cost
            self.falseNegativeCost = false_negative_cost
            self.negativeLabel = negative_label
            # For time keeping        
            self.fitTime = 0
            self.predictTime = 0
            
            self._checkOptions()

        def _checkOptions(self):
            
            if (np.sum([self.ruleLengthCost, self.falseNegativeCost]) > 1):
                print('Works with only one type of cost')
                print('Proceeding with rule length')
                self.falseNegativeCost = False
                self.ruleLengthCost = True

        def _cleanup(self):
            
            self.fittedDTs = {}   
            self.rules = {}      
            self.ruleInfo = {}
            self.labelToInteger = {} 
            self.integerToLabel= {}
            self.missedXvals = None        
            self.numOfMissed = None
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            self.costs = np.empty(shape=(0), dtype=np.float)
                    
        def _getRule(self, fitTree, nodeid):
            
            if (fitTree.tree_.feature[0] == -2): # No rule case
                return Rule()
            left = fitTree.tree_.children_left
            right = fitTree.tree_.children_right
            threshold = fitTree.tree_.threshold
        
            def recurse(left, right, child, returnRule=None):
                if returnRule is None:
                    returnRule = Rule()                
                if child in left: # 'l'
                    parent = np.where(left == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    ub=threshold[parent])
                else: # 'r'
                    parent = np.where(right == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    lb=threshold[parent])                
                returnRule.addClause(clause)
                if parent == 0:
                    return returnRule
                else:
                    return recurse(left, right, parent, returnRule)
        
            retRule = recurse(left, right, nodeid)
        
            return retRule


        def _getInitMatrix(self, X, y, fitTree, treeno):

            if (len(self.cols) == 0):
                col = 0
            else:
                col = max(self.cols) + 1 # Next column        
            y_rules = fitTree.apply(X) # Tells us which sample is in which leaf            
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                leafYvals = y[covers] # y values of the samples in the leaf
                uniqueLabels, counts = np.unique(leafYvals, return_counts=True)
                label = uniqueLabels[np.argmax(counts)] # majority class in the leaf
                labelVector = np.ones(self.K)*(-1/(self.K-1))
                labelVector[self.labelToInteger[label]] = 1
                fillAhat = np.dot(self.vecY[:, covers].T, labelVector)
                self.rows = np.hstack((self.rows, covers))
                self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32)*col))
                self.yvals = np.hstack((self.yvals, np.ones(len(covers), dtype=np.float)*fillAhat))
                if (self.falseNegativeCost):
                    cost = 1.0
                    if (label != self.negativeLabel and self.negativeLabel in leafYvals):
                        cost += np.exp(counts[int(self.negativeLabel)]/np.sum(counts))
                elif (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                else:
                    cost = 1.0
                self.costs = np.append(self.costs, cost)
                self.ruleInfo[col] = (treeno, leafno, label)
                col += 1

            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale


        def _PSPDT(self, X, y, fitTree, treeno, betas, gammas):

            n, col = max(self.rows)+1, max(self.cols)+1
            y_rules = fitTree.apply(X) # Tells us which sample is in which leaf
            noImprovement = True
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                # prepare to check the reduced cost
                aij = np.zeros(n)
                aijhat = np.zeros(n)
                leafYvals = y[covers] # y values of the samples in the leaf
                uniqueLabels, counts = np.unique(leafYvals, return_counts=True)
                label = uniqueLabels[np.argmax(counts)] # majority class in the leaf
                labelVector = np.ones(self.K)*(-1.0/(self.K-1))
                labelVector[self.labelToInteger[label]] = 1
                fillAhat = np.dot(self.vecY[:, covers].T, labelVector)                
                aij[covers] = 1
                aijhat[covers] = fillAhat
                if (self.falseNegativeCost):
                    cost = 1.0
                    if (label != self.negativeLabel and self.negativeLabel in leafYvals):
                        cost += np.exp(counts[int(self.negativeLabel)]/np.sum(counts))
                elif (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                else:
                    cost = 1.0
                cost *= self.wscale
                red_cost = np.dot((((self.K-1.0)/self.K)*aijhat), betas) + \
                    np.dot(aij, gammas) - cost
                if (red_cost > 0): # Only columns with positive reduced costs are added  
                    self.rows = np.hstack((self.rows, covers))
                    self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32)*col))
                    self.yvals = np.hstack((self.yvals, np.ones(len(covers), dtype=np.float)*fillAhat))
                    self.costs = np.append(self.costs, cost)
                    self.ruleInfo[col] = (treeno, leafno, label)
                    col += 1
                    noImprovement = False
                    
            return noImprovement
             

        def _preprocess(self, X, y):
            
            classes, classCounts = np.unique(y, return_counts=True)
            self.majorityClass = classes[np.argmax(classCounts)]
            for i, c in enumerate(classes):
                self.labelToInteger[c] = i
                self.integerToLabel[i] = c
            self.K = len(classes)
            n = len(y)
            self.vscale = 1.0
            self.vecY = np.ones((self.K, n))*(-1/(self.K-1))
            for i, c in enumerate(y):
                self.vecY[self.labelToInteger[c], i] = 1        
            
        def _fillRules(self, weights):
            
            weights = weights/np.max(weights) # Scaled weights
            selectedColumns = np.where(weights > self.threshold)[0] # Selected columns
            weightOrder = np.argsort(-weights[selectedColumns]) # Ordered weights
            orderedColumns = selectedColumns[weightOrder] # Ordered indices
            
            for i, col in enumerate(orderedColumns):
                treeno, leafno, label = self.ruleInfo[col]
                fitTree = self.fittedDTs[treeno]
                if (fitTree.get_n_leaves()==1):
                    self.rules[i] = Rule(label=self.majorityClass,
                                         clauses=[],
                                         weight=weights[col]) # No rule
                else:
                    self.rules[i] = self._getRule(fitTree, leafno)
                    self.rules[i].label = label
                    self.rules[i].weight = weights[col]                
                    self.rules[i]._cleanRule()                

        def _solvePrimal(self, ws0=[], vs0=[]):
            
            if(self.solver == 'glpk'):
                return self._solvePrimalGLPK(ws0=ws0, vs0=vs0)
            elif (self.solver == 'gurobi'):
                return self._solvePrimalGurobi(ws0=ws0, vs0=vs0)
            else:
                print('This solver does not exist')

        def _solvePrimalGLPK(self, ws0=[], vs0=[]):
    
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)        
            
            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale            
            # Variables
            vs = cp.Variable(n, nonneg=True)
            ws = cp.Variable(m, nonneg=True)
            if (len(vs0) > 0):
                vs.value = vs0
            if (len(ws0) > 0):
                ws.value = np.zeros(m)
                ws.value[:len(ws0)] = ws0
            # Primal Model
            primal = cp.Problem(cp.Minimize((np.ones(n)*self.vscale) @ vs + 
                                   self.costs @ ws),
                              [(((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0,
                               A @ ws >= self.eps])
            primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
            betas = primal.constraints[0].dual_value
            gammas = primal.constraints[1].dual_value
            
            return ws.value, vs.value, betas, gammas 
    
        def _solvePrimalGurobi(self, ws0=[], vs0=[]):
    
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)        
            
            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale            
            # Primal Model
            modprimal = gp.Model('RUG Primal')
            modprimal.setParam('OutputFlag', False)
            # variables
            vs = modprimal.addMVar(shape=int(n), name='vs')
            ws = modprimal.addMVar(shape=int(m), name='ws')
            if (len(vs0) > 0):
                vs.setAttr('Start', vs0)
                modprimal.update()
                # print(vs.getAttr(GRB.Attr.Start))
            if (len(ws0) > 0):
                tempws = np.zeros(m)
                tempws[:len(ws0)] = ws0
                ws.setAttr('Start', tempws)
                modprimal.update()
                # print(ws.getAttr(GRB.Attr.Start))
            # objective
            modprimal.setObjective((np.ones(n)*self.vscale) @ vs + 
                                   self.costs @ ws, GRB.MINIMIZE)
            #constraints
            modprimal.addConstr((((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0, 
                                name='Ahat Constraints')
            modprimal.addConstr(A @ ws >= self.eps, name='A Constraints')
            modprimal.optimize()
            betas = np.array(modprimal.getAttr(GRB.Attr.Pi)[:n])
            gammas = np.array(modprimal.getAttr(GRB.Attr.Pi)[n:n+n])
            
            return ws.X, vs.X, betas, gammas


        def _solveDual(self): # Not used in RUG
            
            if(self.solver == 'glpk'):
                return self._solveDualGLPK()
            elif (self.solver == 'gurobi'):
                return self._solveDualGurobi()
            else:
                print('This solver does not exist')
            

        def _solveDualGLPK(self): # Not used in RUG
            
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32) 
            
            n = max(self.rows)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale            
            # Variables
            betas = cp.Variable(n, nonneg=True)
            gammas = cp.Variable(n, nonneg=True)
            # Objective
            dual = cp.Problem(cp.Maximize(np.ones(n) @ betas + (np.ones(n)*self.eps) @ gammas),
                              [(((self.K - 1.0)/self.K)*Ahat.T) @ betas + A.T @ gammas <= self.costs,
                               betas <= self.vscale])
            
            dual.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})

            ws = dual.constraints[0].dual_value
            
            return betas.value, gammas.value, ws


        def _solveDualGurobi(self): # Not used in RUG
            
            Ahat = csr_matrix((self.yvals, (self.rows, self.cols)), dtype=np.float)
            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32) 
            
            n = max(self.rows)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale              
            # Dual Model
            moddual = gp.Model('RUG Dual')
            moddual.setParam('OutputFlag', False)
            # variables
            betas = moddual.addMVar(shape=int(n), ub=self.vscale, name='betas')
            gammas = moddual.addMVar(shape=int(n), name='gammas')        
            #objective
            moddual.setObjective(np.ones(n) @ betas + (np.ones(n)*self.eps) @ gammas, 
                                 GRB.MAXIMIZE)        
            #constraints
            moddual.addConstr((((self.K - 1.0)/self.K)*Ahat.T) @ betas + A.T @ gammas 
                              <= self.costs)
            moddual.optimize()
            ws = np.array(moddual.getAttr(GRB.Attr.Pi)) # w values
            
            return betas.X, gammas.X, ws
        
        def printRules(self, indices=[]):
            
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING! printRules() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                if (rule == 'NR'):
                    print('==> No Rule: Set Majority Class')
                else:
                    rule.printRule()
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
    
        def printWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: printWeights() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
                
        def getWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: getWeights() ###\n')
                print('Do not have that many rules')                
                return None 
            
            return [self.rules[indx].weight for indx in indices]    
                
        def predict(self, X, indices=[]):       
            
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: predict() ###\n')
                print('Do not have that many rules')
                return None  
    
            self.missedXvals = []        
            self.numOfMissed = 0
            
            startTime = time.time()
            # TODO: Can be done in parallel
            returnPrediction = []
            for x0 in X:
                sumClassWeights = np.zeros(self.K)
                for indx in indices:
                    rule = self.rules[indx]
                    if (rule != 'NR'):
                        if(rule.checkRule(x0)):
                            lab2int = self.labelToInteger[rule.label]
                            sumClassWeights[lab2int] += rule.weight
                
                if (np.sum(sumClassWeights) == 0):
                    # Unclassified test sample
                    self.numOfMissed += 1
                    self.missedXvals.append(x0)
                    # Assigned to a class with the initial DT
                    getClass = self.fittedDTs[0].predict(x0.reshape(1, -1))[0]
                    returnPrediction.append(getClass)
                else:
                    sel_label_indx = np.argmax(sumClassWeights)
                    int2lab = self.integerToLabel[sel_label_indx]
                    returnPrediction.append(int2lab)
    
            endTime = time.time()
            self.predictTime = endTime - startTime
            
            return returnPrediction
    
        def getAvgRuleLength(self):
            
            return np.mean([rule.length() for rule in self.rules.values()])
            
        def getNumOfRules(self):
            
            return len(self.rules)
    
        def getNumOfMissed(self):
            
            return self.numOfMissed
    
        def getFitTime(self):
            
            return self.fitTime
    
        def getPredictTime(self):
            
            return self.predictTime
        
        def fit(self, X, y):
            
            if (len(self.cols) != 0):
                self._cleanup()

            startTime = time.time()
            
            treeno = 0
            DT = DecisionTreeClassifier(max_depth=self.maxDepth,
                                        random_state=self.randomState)
            fitTree = DT.fit(X, y)
            self.fittedDTs[treeno] = copy.deepcopy(fitTree)
            self._preprocess(X, y)
            self._getInitMatrix(X, y, fitTree, treeno)
            # TODO: Her seferinde yeni model kurmadan da olmalı aslında,
            # aynı model gitmez mi acaba? Gurobi'ye bakmalı
            ws, vs, betas, gammas = self._solvePrimal()
            # Column generation
            for cg in range(self.maxRMPcalls):        
                treeno += 1
                fitTree = DT.fit(X, y, sample_weight=betas) # Use duals as weights                  
                self.fittedDTs[treeno] = copy.deepcopy(fitTree)
                noImprovement = self._PSPDT(X, y, fitTree, treeno, betas, gammas)
                if (noImprovement):
                    break
                ws, vs, betas, gammas = self._solvePrimal(ws0=ws, vs0=vs)
            self._fillRules(ws)
            
            endTime = time.time()
            
            self.fitTime = endTime - startTime
