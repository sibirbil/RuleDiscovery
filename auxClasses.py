"""
@author: sibirbil
"""
import numpy as np

class SklearnEstimator:
    '''
    This base class is dummy, just for guideline. Sklearn does not
    provide a base class that has both fit and predict
    '''
    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError('Needs to implement fit(X, y)')

    def predict(self, X0: np.array):
        raise NotImplementedError('Needs to implement predict(X, y)')

class Coefficients:
    
        def __init__(self, yvals=np.empty(shape=(0), dtype=np.float64),
                     rows=np.empty(shape=(0), dtype=np.int32),
                     cols=np.empty(shape=(0), dtype=np.int32),
                     costs=np.empty(shape=(0), dtype=np.float64)):

            self.yvals = yvals
            self.rows = rows
            self.cols = cols
            self.costs = costs
            
        def _cleanup(self):
            self.yvals = np.empty(shape=(0), dtype=np.float64)
            self.rows=np.empty(shape=(0), dtype=np.int32)
            self.cols=np.empty(shape=(0), dtype=np.int32)
            self.costs=np.empty(shape=(0), dtype=np.float64)

class Clause:
    
    def __init__(self, feature=None, ub=np.inf, lb=-np.inf):
        self.feature = feature # used feature
        self.ub = ub # upper bound
        self.lb = lb # lower bound
        
    def _toText(self):
        returnString = ''
        if (self.lb != -np.inf and self.ub != np.inf):
            returnString += '{:.2f}'.format(self.lb) + ' < ' + \
                'x[' + str(self.feature) + ']' + ' <= ' + '{:.2f}'.format(self.ub)
        else:
            if (self.lb != -np.inf):
                returnString += 'x[' + str(self.feature) + ']' + \
                    ' > ' + '{:.2f}'.format(self.lb)
            if (self.ub != np.inf):
                returnString += 'x[' + str(self.feature) + ']' + \
                    ' <= ' + '{:.2f}'.format(self.ub)
        return returnString
    
    def printClause(self):
        print(self._toText())
    
    def _checkArray(self, X):
        # Converts input to a double array
        if (type(X) != np.ndarray):
            X = np.array(X)
        if (len(X.shape) == 0):
            X = np.array([[X]])
        if (len(X.shape) == 1):
            X = np.array([X])
        return X

    def checkClause(self, X):
        X = self._checkArray(X)
        n = len(X)
        returnList = np.array(n * [False])
        for indx, x0 in enumerate(X):
            val = x0[self.feature]
            if (self.lb != -np.inf and self.ub != np.inf):
                if (self.lb < val and val <= self.ub):
                    returnList[indx] = True
            else:
                if (self.lb != -np.inf):
                    if (self.lb < val):
                        returnList[indx] = True                    
                if (self.ub != np.inf):
                    if (val <= self.ub):
                        returnList[indx] = True
        return returnList

class Rule:
    
    def __init__(self, label=None, clauses=[], weight=None):
        self.label = label # class
        self.clauses = clauses # clauses
        self.weight = weight # weight
        self.cleaned = False

    def _toText(self):
        returnString = ''
        for clause in self.clauses:
            returnString += clause._toText() + '\n'
        if (len(self.clauses) == 0): # No Rule
            returnString += '# No Rule #'
        return returnString
    
    def addClause(self, clause):
        if (self.clauses == []):
            self.clauses = [clause]
        else:
            self.clauses.append(clause)
            
    def length(self):
        if (~self.cleaned):
            self._cleanRule()
            self.cleaned = True
            
        return len(self.clauses)
        
    def printRule(self):
        if (~self.cleaned):
            self._cleanRule()
            self.cleaned = True        

        print(self._toText())

    def _checkArray(self, X):
        # Converts input to a double array        
        if (type(X) != np.ndarray):
            X = np.array(X)
        if (len(X.shape) == 0):
            X = np.array([[X]])
        if (len(X.shape) == 1):
            X = np.array([X])
        return X

    def checkRule(self, X):
        X = self._checkArray(X)
        n = len(X)
        returnList = np.array(n * [False])
        trueSampleIndices = np.array(range(n))
        for clause in self.clauses:
            remOnes = clause.checkClause(X[trueSampleIndices, :])
            trueSampleIndices = trueSampleIndices[remOnes]
            if (trueSampleIndices.size == 0):
                break
            
        returnList[trueSampleIndices] = True

        return returnList
    
    def _cleanRule(self):
        # TODO: Probably can be done more efficiently
        # but overhead is negligible
        if (~self.cleaned):
            n = len(self.clauses)
            remainingRules = set(range(n))
            for i in range(n-1):
                ci = self.clauses[i]
                for j in range(i+1, n):
                    cj = self.clauses[j]
                    if (ci.feature == cj.feature):
                        ci.ub = np.min([ci.ub, cj.ub])
                        ci.lb = np.max([ci.lb, cj.lb])
                        if (j in remainingRules):
                            remainingRules.remove(j)
            newClauses = [self.clauses[i] for i in remainingRules]
            self.clauses = newClauses
            self.cleaned = True