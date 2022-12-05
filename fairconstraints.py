import numpy as np
import pandas as pd
import itertools
from statistics import mean

def setEqualOpportunity(X, y, classLabel, group_pair):
    n = len(y)
    temp = np.zeros(n)

    # Change all entries with the occuring classlabel to 1 to count in the summation
    for idx in range(0,n):
        if y[idx] == classLabel:
            temp[idx] = 1

    dfX = pd.DataFrame(X)
    sensitive_attribute = dfX[0] # sensitive attribute should be first column of data set

    group1 = temp.copy()
    group2 = temp.copy()

    # Change all entries with the occuring group to 1 for Pk,1 and the complement for Pk,0
    for idx in range(0,n):
        if temp[idx] == 1:
            if sensitive_attribute[idx] == group_pair[0]: # does not work for continuous values
                group2[idx] = 0
            else:
                group1[idx] =0

    return group1, group2

def setEqualOverallMistreatment(X, y, group_pair):
    n = len(y)
    setI = np.ones(len(y))

    dfX = pd.DataFrame(X)
    sensitive_attribute = dfX[0]

    group1 = setI.copy()
    group2 = setI.copy()

    for idx in range(0,n):
        if sensitive_attribute[idx] == group_pair[0]:
            group2[idx] = 0
        else:
            group1[idx] = 0

    return group1, group2

def create_setsPI(X,y, groups, metric):
    pairs = list(itertools.combinations(groups,2))
    setP_pairs = []
    setI_pairs = []
    setI = [] # this won't be used as fairnes constraints won't be added
    classes = pd.unique(y)
    classes.sort()

    for idx in range(0, len(classes)):
        classLabel = classes[idx]
        for pair in pairs:
            if metric == None:
                group1 = np.zeros(len(y))
                group2 = np.zeros(len(y))
                setI.append([group1, group2])
                constraintSet = setI
           
            if metric == "dmc":
                group1, group2 = setEqualOpportunity(X, y, classLabel, pair)
                setP_pairs.append([group1,group2])
                constraintSet = setP_pairs
              
            if metric == "odm":
                group1, group2 = setEqualOverallMistreatment(X,y, pair)
                setI_pairs.append([group1, group2])
                constraintSet = setI_pairs
            
    return constraintSet, pairs

def fairnessEvaluation(y_actual, y_pred, setsP, classes, pairs):
    if len(classes)>2: # for Multi-Class
        unfairness_mat = []
        for pair in setsP: 
            set_group1 = pair[0]
            set_group2 = pair[1]
            setsize_group1 = int(sum(set_group1))
            setsize_group2 = int(sum(set_group2))


            if setsize_group1==0:
                alpha=0
            elif setsize_group2==0:
                alpha=0
            else:
                u1 = np.zeros(len(y_actual)) 
                u2 = np.zeros(len(y_actual))

                
                # MISSCLASSIFCATION COUNTING VECTOR FOR GROUP G=1
                for idx in range(0, len(set_group1)): 
                    if set_group1[idx] == 1: 
                        if y_actual[idx] != y_pred[idx]: 
                            u1[idx] = 1 
                
                # MISCLASSIFCAITON COUTNING VECTOR FOR GROUP G'=2
                for idx in range(0, len(set_group2)): 
                    if set_group2[idx] == 1:
                        if y_actual[idx] != y_pred[idx]:
                            u2[idx] = 1

                # print("number of misclassifications in i in Pkg: ", sum(u1))
                # print("number of misclassifications in i in Pkg': ", sum(u2))
                # print("set size group1: ", setsize_group1)
                # print("set size group2: ", setsize_group2)

                alpha = abs((1/setsize_group1)*sum(u1)-(1/setsize_group2)*sum(u2))
           
            unfairness_mat.append(alpha)
        unfairness = unfairnessLevel_multiclass(unfairness_mat, classes, pairs)

    else: # For Binary Class
        # Assume that the second class of binary class is the advantageous one, so 0 = negative, 1 = positive
        pair = setsP[1] 
        set_group1 = pair[0]
        set_group2 = pair[1]
        setsize_group1 = int(sum(set_group1))
        setsize_group2 = int(sum(set_group2))

        # print(setsize_group1)
        # print(setsize_group2)
 
        if setsize_group1==0:                
            alpha=0
        elif setsize_group2==0:
            alpha=0
        else:
            u1 = np.zeros(len(y_actual)) 
            u2 = np.zeros(len(y_actual))

            # MISSCLASSIFCATION COUNTING VECTOR FOR GROUP G=1
            for idx in range(0, len(set_group1)): 
                if set_group1[idx] == 1: 
                    if y_actual[idx] != y_pred[idx]:
                        u1[idx] = 1 
                
            # MISCLASSIFCAITON COUTNING VECTOR FOR GROUP G'=2
            for idx in range(0, len(set_group2)): 
                if set_group2[idx] == 1:
                    if y_actual[idx] != y_pred[idx]:
                        u2[idx] = 1

            alpha = abs((1/setsize_group1)*sum(u1)-(1/setsize_group2)*sum(u2))
        unfairness = alpha
        #print(unfairness)

    return unfairness

def unfairnessLevel_multiclass(unfairness_array, classes, group_pairs):
    sizeK = len(classes)
    sizeD = len(group_pairs)

    unfairness_helparray = unfairness_array.copy()
    maxGG = []
        
    while len(unfairness_helparray)>0:
        grouparray = unfairness_helparray[0:sizeD]
        maxval = max(grouparray)
        maxGG.append(maxval)
        temp_array = unfairness_helparray[sizeD:]
        unfairness_helparray=temp_array
        
        alpha_k = mean(maxGG)
    return alpha_k