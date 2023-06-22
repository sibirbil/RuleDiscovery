"""
@author: sibirbil
"""
import copy
import time
import warnings
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from auxClasses import SklearnEstimator, Coefficients, Clause, Rule

class RUXG(BaseEstimator, SklearnEstimator):
    
    def __init__(self):
        
        self.fitTime = 0
        self.predictTime = 0
            
    def _initialize(self):
        
        self.fittedDTs = {}
        self.initNumOfRules = 0
        self.rules = {}      
        self.ruleInfo = {}        
        self.missedXvals = None        
        self.numOfMissed = None
        self.rulesPerSample = None
        self.ruleLengthPerSample = None
        self.normConst = 1.0
        # Coefficients (A, Abar & costs) are used as CSR sparse matrices
        self._coeffs = Coefficients()
        
    def _checkOptions(self):
        
        if (type(self) == RUXClassifier):

            if (self.trained_ensemble == None):
                raise ValueError('One ensemble learning method (RF, ADA or GB) should be provided.')
                
            try:
                check_is_fitted(self.trained_ensemble)
            except NotFittedError as e:
                raise ValueError('Ensemble learning method should be fitted first to use in RUX.')
            

            for treeno, fitTree in enumerate(self.trained_ensemble.estimators_):
                if (type(self.trained_ensemble) == GradientBoostingClassifier or
                    type(self.trained_ensemble) == GradientBoostingRegressor):
                    self.initNumOfRules += fitTree[0].get_n_leaves()
                    self.fittedDTs[treeno] = fitTree[0]                
                else:
                    self.initNumOfRules += fitTree.get_n_leaves()
                    self.fittedDTs[treeno] = fitTree
                    
    def _cleanup(self):
        
        self.labelToInteger = {} 
        self.integerToLabel= {}

        self.fittedDTs = {}   
        self.rules = {}      
        self.ruleInfo = {}
        self.missedXvals = None
        self.numOfMissed = None
        self.rulesPerSample = None
        self.ruleLengthPerSample = None
        self._coeffs._cleanup()    

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

        if (len(self._coeffs.cols) == 0):
            col = 0
        else:
            col = max(self._coeffs.cols) + 1 # Next column        
        y_rules = fitTree.apply(X) # tells us which sample is in which leaf            
        for leafno in np.unique(y_rules):
            covers = np.where(y_rules == leafno)[0]
            leafYvals = y[covers] # y values of the samples in the leaf
            uniqueLabels, counts = np.unique(leafYvals, return_counts=True)
            label = uniqueLabels[np.argmax(counts)] # majority class in the leaf
            labelVector = np.ones(self.K)*(-1/(self.K-1))
            labelVector[self.labelToInteger[label]] = 1
            fillAhat = np.dot(self.vecY[:, covers].T, labelVector)
            self._coeffs.rows = np.hstack((self._coeffs.rows, covers))
            self._coeffs.cols = np.hstack((self._coeffs.cols, np.ones(len(covers), dtype=np.int32)*col))
            self._coeffs.yvals = np.hstack((self._coeffs.yvals, np.ones(len(covers), dtype=np.float64) * fillAhat))
            if (self.rule_length_cost):
                tempRule = self._getRule(fitTree, leafno)
                cost = tempRule.length()
            else:
                cost = 1.0
            self._coeffs.costs = np.append(self._coeffs.costs, cost)
            self.ruleInfo[col] = (treeno, leafno, label)
            col += 1
        
        self.normConst = 1.0/np.max(self._coeffs.costs)
        
    def _getMatrices(self, X, y):
        
        if (type(self) == RUXClassifier):
        
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
        self.vecY = np.ones((self.K, n))*(-1/(self.K-1))
        for i, c in enumerate(y):
            self.vecY[self.labelToInteger[c], i] = 1

    def _fillRules(self, weights):
        
        # The weights are scaled only for classification
        if (np.max(weights > 1.0e-6)):
            weights = weights/np.max(weights) # Scaled weights
         
        selectedColumns = np.where(weights > self.threshold)[0] # selected columns
        weightOrder = np.argsort(-weights[selectedColumns]) # ordered weights
        orderedColumns = selectedColumns[weightOrder] # ordered indices
        
        for i, col in enumerate(orderedColumns):
            treeno, leafno, label = self.ruleInfo[col]
            fitTree = self.fittedDTs[treeno]
            if (fitTree.get_n_leaves()==1):
                self.rules[i] = Rule(label=self.majorityClass,
                                        clauses=[], weight=weights[col]) # no rule
            else:
                self.rules[i] = self._getRule(fitTree, leafno)
                self.rules[i].label = label
                self.rules[i].weight = weights[col]                
                self.rules[i]._cleanRule()

    def _solvePrimal(self, y=None, ws0=[], vs0=[], groups=None):
        
         # TODO: Pyomo
        
        if(self.solver == 'glpk'):
            return self._solvePrimalGLPK(y=y, ws0=ws0, vs0=vs0, groups=groups)
        elif (self.solver == 'gurobi'):
            return self._solvePrimalGurobi(y=y, ws0=ws0, vs0=vs0, groups=groups)
        else:
            raise ValueError('Solver {0} does not exist'.format(self.solver))

    def _solvePrimalGLPK(self, y=None, ws0=[], vs0=[], groups=None):

        Ahat = csr_matrix((self._coeffs.yvals, (self._coeffs.rows, self._coeffs.cols)), dtype=np.float64)

        n, m = max(self._coeffs.rows)+1, max(self._coeffs.cols)+1       
        # Variables
        vs = cp.Variable(n, nonneg=True)
        ws = cp.Variable(m, nonneg=True)
        if (len(vs0) > 0):
            vs.value = vs0
        if (len(ws0) > 0):
            ws.value = np.zeros(m)
            ws.value[:len(ws0)] = ws0
        # Primal Model
        primal = cp.Problem(cp.Minimize(np.ones(n) @ vs + 
                                (self._coeffs.costs * self.pen_par * self.normConst) @ ws),
                            [(((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0])
        
        # Fairness constraints
        if self.fair_metric==None:
            primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
        
        if self.fair_metric=='dmc' or self.fair_metric=='odm':
            for pair in groups:
                setgroup1 = pair[0]
                setgroup2 = pair[1]

                amount1 = sum(setgroup1)
                amount2 = sum(setgroup2)

                if amount1==0 or amount2==0:
                    continue
                else:
                    fairness_constraints = [(((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0]
                    fairness_constraints.append((((1.0/amount1)*setgroup1) - ((1/amount2)*setgroup2)) @ vs <= self.fair_eps)
                    fairness_constraints.append((((1.0/amount2)*setgroup2) - ((1/amount1)*setgroup1)) @ vs <= self.fair_eps)
                    primal = cp.Problem(cp.Minimize(np.ones(n) @ vs + 
                                    (self._coeffs.costs * self.pen_par * self.normConst) @ ws),fairness_constraints)
                primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})

        betas = primal.constraints[0].dual_value
        
        return ws.value, vs.value, betas

    def _solvePrimalGurobi(self, y=None, ws0=[], vs0=[], groups=None):

        Ahat = csr_matrix((self._coeffs.yvals, (self._coeffs.rows, self._coeffs.cols)), dtype=np.float64)

        n, m = max(self._coeffs.rows)+1, max(self._coeffs.cols)+1    
        # Primal Model
        modprimal = gp.Model('RUXG Primal')
        modprimal.setParam('OutputFlag', False)
        # Variables
        vs = modprimal.addMVar(shape=int(n), name='vs')
        ws = modprimal.addMVar(shape=int(m), name='ws')
        if (len(vs0) > 0):
            vs.setAttr('Start', vs0)
            modprimal.update()
        if (len(ws0) > 0):
            tempws = np.zeros(m)
            tempws[:len(ws0)] = ws0
            ws.setAttr('Start', tempws)
            modprimal.update()
        # Objective
        modprimal.setObjective(np.ones(n) @ vs + 
                                (self._coeffs.costs * self.pen_par * self.normConst) @ ws, GRB.MINIMIZE)
        # Constraints
        modprimal.addConstr((((self.K - 1.0)/self.K)*Ahat) @ ws + vs >= 1.0, name='Ahat Constraints')

        # Fairness constraints
        if self.fair_metric==None:
            modprimal.optimize()

        if self.fair_metric == 'dmc' or self.fair_metric == 'odm':
            for pair in groups: # groups is a pair of groups
                setgroup1 = pair[0]
                setgroup2 = pair[1]

                amount1 = sum(setgroup1)
                amount2 = sum(setgroup2)

                if amount1==0 or amount2==0:
                    continue
                else:
                    modprimal.addConstr((((1.0/amount1)*setgroup1) - ((1/amount2)*setgroup2)) @ vs <= self.fair_eps, name='Fairness constraints 1')
                    modprimal.addConstr((((1.0/amount2)*setgroup2) - ((1/amount1)*setgroup1)) @ vs <= self.fair_eps, name='Fairness constraints 2')
            modprimal.optimize()

        betas = np.array(modprimal.getAttr(GRB.Attr.Pi)[:n])        
        
        return ws.X, vs.X, betas
    
    def print_rules(self, indices=[]):
        
        if (len(indices) == 0):
            indices = self.rules.keys()
        elif (np.max(indices) > len(self.rules)):
            warnings.warn('\n There are only {0} rules'.format(len(self.rules)))                
            return
        
        for indx in indices:
            rule = self.rules[indx]
            print('RULE %d:' % (indx))
            if (rule.length() == 0):
                print('==> No Rule: Set Majority Class')
            else:
                rule.printRule()
            print('Class: %.0f' % rule.label)
            print('Scaled rule weight: %.4f\n' % rule.weight)

    def print_rules_for_instances(self, IDs, x_id_to_rule_ids_dict, colnames):
        for x0_id in IDs:
            print('Rules for the instance:\n')
            rule_ids = x_id_to_rule_ids_dict[x0_id]
            self.print_rules(feature_names=colnames, indices=rule_ids)
            print('\n \n')  

    def print_weights(self, indices=[]):

        if (len(indices) == 0):
            indices = self.rules.keys()
        elif (np.max(indices) > len(self.rules)):
            warnings.warn('\n There are only {0} rules'.format(len(self.rules)))                 
            return
        
        for indx in indices:
            rule = self.rules[indx]
            print('RULE %d:' % (indx))
            print('Value: %.0f' % rule.label)
            print('Scaled rule weight: %.4f\n' % rule.weight)
            
    def get_weights(self, indices=[]):

        if (len(indices) == 0):
            indices = self.rules.keys()
        elif (np.max(indices) > len(self.rules)):
            warnings.warn('\n There are only {0} rules'.format(len(self.rules)))  
            return None 
        
        return [self.rules[indx].weight for indx in indices]   
    
    def get_avg_rule_length(self):
        
        return np.mean([rule.length() for rule in self.rules.values()])
        
    def get_num_of_rules(self):
        
        return len(self.rules)

    def get_init_num_of_rules(self):
        
        return self.initNumOfRules

    def get_num_of_missed(self):
        
        return self.numOfMissed
    
    def get_avg_num_rules_per_sample(self):
        
        return np.mean(self.rulesPerSample)

    def get_avg_rule_length_per_sample(self):

        return np.mean(self.ruleLengthPerSample)

    def get_fit_time(self):
        
        return self.fitTime

    def get_predict_time(self):
        
        return self.predictTime

    def predict(self, X, indices=[]):       
        
        if (self.fittedDTs == {}):
            raise ValueError('You need to fit the RUG model first')
        
        if (len(indices) == 0):
            indices = self.rules.keys()
        elif (np.max(indices) > len(self.rules)):
            warnings.warn('\n There are only {0} rules'.format(len(self.rules)))
            return None  

        self.missedXvals = []        
        self.numOfMissed = 0
        self.rulesPerSample = np.zeros(len(X))
        self.ruleLengthPerSample = np.zeros(len(X))
        
        startTime = time.time()
        returnPrediction = []
        for sindx, x0 in enumerate(X):
            sumClassWeights = np.zeros(self.K)
            rule_lengths = []
            for indx in indices:
                rule = self.rules[indx]
                if(rule.checkRule(x0)):
                    lab2int = self.labelToInteger[rule.label]
                    sumClassWeights[lab2int] += rule.weight
                    self.rulesPerSample[sindx] += 1.0
                    rule_lengths.append(rule.length())
            if len(rule_lengths) > 0:
                self.ruleLengthPerSample = np.mean(rule_lengths)
            
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

class RUXClassifier(RUXG, ClassifierMixin):
    
    def __init__(self, trained_ensemble=None, pen_par=1.0,
                 threshold=1.0e-6, rule_length_cost=False, 
                 solver='gurobi', fair_eps=1.0, fair_metric=None, random_state=None):
        
        self._initialize()
        self.trained_ensemble = trained_ensemble
        self.pen_par = pen_par
        self.threshold = threshold
        self.solver = solver
        self.fair_eps = fair_eps
        self.fair_metric = fair_metric
        self.random_state = random_state
        self.rng_ = np.random.default_rng(self.random_state)
        self.K = None # number of classes
        self.labelToInteger = {} # mapping classes to integers
        self.integerToLabel= {} # mapping integers to classes
        self.vecY = None
        self.majorityClass = None
        self.rule_length_cost = rule_length_cost
        # Classifier type
        self._checkOptions()

    def fit(self, X, y, groups=None):
        
        if (len(self._coeffs.cols) != 0):
            self._cleanup()

        if (self.fair_metric == None):
            groups = None
        elif(groups == None):
            warnings.warn('Groups should be provided. Setting fairness metric to None.')  
            self.fair_metric = None
    
        startTime = time.time()
        
        self._preprocess(X, y)
        self._getMatrices(X, y)

        ws = self._solvePrimal(groups=groups)[0]
        
        self._fillRules(ws)
        
        endTime = time.time()
        
        self.fitTime = endTime - startTime
        
        return self
        
class RUGClassifier(RUXG, ClassifierMixin):
    '''
    Parameters
    ----------
        pen_par: 
    '''
    def __init__(self, pen_par=1.0, threshold=1.0e-6,
                 max_depth=None, max_RMP_calls=30, rule_length_cost=False,
                 solver='gurobi', fair_eps=1.0, fair_metric=None, random_state=None):
        
        self._initialize()
        self.pen_par = pen_par
        self.threshold = threshold
        self.solver = solver
        self.fair_eps = fair_eps
        self.fair_metric = fair_metric        
        self.random_state = random_state
        self.rng_ = np.random.default_rng(self.random_state)
        self.K = None # number of classes
        self.labelToInteger = {} # mapping classes to integers
        self.integerToLabel= {} # mapping integers to classes
        self.vecY = None
        self.majorityClass = None
        self.max_depth = max_depth
        self.max_RMP_calls = max_RMP_calls
        self.rule_length_cost = rule_length_cost

    def _PSPDT(self, X, y, fitTree, treeno, betas):

        n, col = len(X), max(self._coeffs.cols)+1
        y_rules = fitTree.apply(X) # tells us which sample is in which leaf
        noImprovement = True
        for leafno in np.unique(y_rules):
            covers = np.where(y_rules == leafno)[0]
            # Prepare to check the reduced cost
            aijhat = np.zeros(n)
            leafYvals = y[covers] # y values of the samples in the leaf
            uniqueLabels, counts = np.unique(leafYvals, return_counts=True)
            label = uniqueLabels[np.argmax(counts)] # majority class in the leaf
            labelVector = np.ones(self.K)*(-1.0/(self.K-1))
            labelVector[self.labelToInteger[label]] = 1
            fillAhat = np.dot(self.vecY[:, covers].T, labelVector)                
            aijhat[covers] = fillAhat
            if (self.rule_length_cost):
                tempRule = self._getRule(fitTree, leafno)
                cost = tempRule.length()
            else:
                cost = 1.0
                
            red_cost = np.dot((((self.K-1.0)/self.K)*aijhat), betas) - (cost * self.pen_par * self.normConst)
            if (red_cost > 0): # only columns with proper reduced costs are added  
                self._coeffs.rows = np.hstack((self._coeffs.rows, covers))
                self._coeffs.cols = np.hstack((self._coeffs.cols, np.ones(len(covers), dtype=np.int32)*col))
                self._coeffs.yvals = np.hstack((self._coeffs.yvals, np.ones(len(covers), dtype=np.float64) * fillAhat))
                self._coeffs.costs = np.append(self._coeffs.costs, cost)
                self.ruleInfo[col] = (treeno, leafno, label)
                col += 1
                noImprovement = False
                
        return noImprovement
    
    def fit(self, X, y, groups=None):
            
        if (len(self._coeffs.cols) != 0):
            self._cleanup()

        if (self.fair_metric == None):
            groups = None
        elif(groups == None):
            warnings.warn('Groups should be provided. Setting fairness metric to None.')  
            self.fair_metric = None
        
        startTime = time.time()
        
        treeno = 0
        DT = DecisionTreeClassifier(max_depth=self.max_depth,
                                    random_state=self.rng_.integers(np.iinfo(np.int16).max))
        fitTree = DT.fit(X, y)
        self.fittedDTs[treeno] = copy.deepcopy(fitTree)
        self._preprocess(X, y)
        self._getMatrix(X, y, fitTree, treeno)
        ws, vs, betas = self._solvePrimal(groups=groups)
        # Column generation
        for _ in range(self.max_RMP_calls):
            treeno += 1
            DT = DecisionTreeClassifier(max_depth=self.max_depth,
                                        random_state=self.rng_.integers(np.iinfo(np.int16).max))
            fitTree = DT.fit(X, y, sample_weight=betas) # use duals as weights                  
            self.fittedDTs[treeno] = copy.deepcopy(fitTree)
            noImprovement = self._PSPDT(X, y, fitTree, treeno, betas)
            if (noImprovement):
                break
            ws, vs, betas = self._solvePrimal(ws0=ws, vs0=vs, groups=groups)
        self._fillRules(ws)
        
        endTime = time.time()
        
        self.fitTime = endTime - startTime
        
        return self
