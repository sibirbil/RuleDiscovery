import numpy as np
import pandas as pd
import itertools
from statistics import mean

def setDMC(X, y, classLabel, group_pair):
    '''
    Creates two vectors, one for group1 and one for group2, that has element = 1 if the sample has label==classLabel AND belongs to the group
    '''
    n = len(y)
    temp = np.zeros(n)

    # Change all entries with the occuring classlabel to 1 to count in the summation
    for idx in range(0,n):
        if y[idx] == classLabel:
            temp[idx] = 1

    dfX = pd.DataFrame(X)
    sensitive_attribute = dfX[0] # sensitive attribute should be first column of data set, create vector indicating grouplabel of each sample

    group1 = temp.copy()
    group2 = temp.copy()

    # Given a grouppair, change all entries with the occuring group to 1 for Pk,1 and the complement for Pk,0
    for idx in range(0,n):
        if temp[idx] == 1:
            if sensitive_attribute[idx] == group_pair[0]: # does not work for continuous values
                group2[idx] = 0 
            else:
                group1[idx] = 0

    return group1, group2

def setODM(X, y, group_pair):
    '''
    Creates vector for group1 and group2 with element=1 if the sample belongs to the corresponding group (1 or 2)
    '''
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
    '''
    Create sets I for each class k and pair of groups.
    '''
    pairs = list(itertools.combinations(groups,2))
    setP_pairs = [] #this is for DMC, so based on class k and grouppair (g,g')
    setI_pairs = [] #this is for ODM, so only based on grouppair (g,g')
    setI = [] # this won't be used as fairnes constraints won't be added
    classes = pd.unique(y)
    classes.sort()

    if len(classes) > 2:
        for idx in range(0, len(classes)):
            classLabel = classes[idx]
            for pair in pairs:
                if metric == None:
                    group1 = np.zeros(len(y))
                    group2 = np.zeros(len(y))
                    setI.append([group1, group2])
                    constraintSet = setI
            
                if metric == "dmc":
                    group1, group2 = setDMC(X, y, classLabel, pair)
                    setP_pairs.append([group1,group2])
                    constraintSet = setP_pairs
                
                if metric == "odm":
                    group1, group2 = setODM(X,y, pair)
                    setI_pairs.append([group1, group2])
                    constraintSet = setI_pairs

    if len(classes) == 2:          
        if metric == "odm":
            for pair in pairs:
                group1, group2 = setODM(X,y, pair)
                setI_pairs.append([group1, group2])
                constraintSet = setI_pairs
        
        if metric == "EqOpp":
            classLabel = 1
            for pair in pairs:
                group1, group2 = setDMC(X, y, classLabel, pair)
                setP_pairs.append([group1,group2])
                constraintSet = setP_pairs
        
        if metric == "dmc":
            for idx in range(0, len(classes)):
                for pair in pairs:
                    classLabel = classes[idx]
                    group1, group2 = setDMC(X, y, classLabel, pair)
                    setP_pairs.append([group1,group2])
                    constraintSet = setP_pairs



    #     for pair in pairs:
    #         if metric == None:
    #             group1 = np.zeros(len(y))
    #             group2 = np.zeros(len(y))
    #             setI.append([group1, group2])
    #             constraintSet = setI
            
    #         if metric == "odm":
    #             print('Binary-class, ODM')
    #             group1, group2 = setODM(X,y, pair)
    #             setI_pairs.append([group1, group2])
    #             constraintSet = setI_pairs
            
    #         if metric == "EqOpp":
    #             classLabel = 1 # We only create the set Ikg and Ikg' for k = 1 (the positive class)
    #             group1, group2 = setDMC(X, y, classLabel, pair)
    #             setP_pairs.append([group1,group2])
    #             constraintSet = setP_pairs
            
    #         if metric == "dmc":
    #         # We create the set Ikg and Ikg' for k = 0,1  (the positive class AND negative class). Note that this is the same as Equalized Odds
    #             print('Binary-class, DMC/Equalized Odds')
    #             classLabel = 0 
    #             group1, group2 = setDMC(X, y, classLabel, pair)
    #             setP_pairs.append([group1,group2])

    #             classLabel = 1 #second class should be the advantageous one for calculating scores.
    #             group1, group2 = setDMC(X, y, classLabel, pair)
    #             setP_pairs.append([group1,group2])
                
    #             constraintSet = setP_pairs
                
    return constraintSet, pairs

def fairnessEvaluation(y_actual, y_pred, setsP, classes, pairs): #Only for multi-class!
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
        #pair = setsP[1] #setsP = [(I0,I1) (label=0), (I0,I1) (label=1)], so setsP[1] are the samples with postive label 1 belonging to group 0 and group 1 
        pair = setsP[0] #setsP = [ (I0, I1) ].  Label doesn't matter in ODM
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

    return unfairness

def binary_EqOdds(y_actual, y_pred, setsP, classes, pairs):
    if len(classes)==2: # For Binary Class
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
            u1 = np.zeros(len(y_actual)) #misclassification counting vectors for group 1 and 2 below.
            u2 = np.zeros(len(y_actual))

            # MISSCLASSIFCATION COUNTING VECTOR FOR GROUP G=1
            for idx in range(0, len(set_group1)): 
                if set_group1[idx] == 1: #if in group 1
                    if y_actual[idx] != y_pred[idx]: #if misclassified
                        u1[idx] = 1  #set counting vector idx to 1
                
            # MISCLASSIFCAITON COUTNING VECTOR FOR GROUP G'=2
            for idx in range(0, len(set_group2)): 
                if set_group2[idx] == 1:
                    if y_actual[idx] != y_pred[idx]:
                        u2[idx] = 1

            alpha = abs((1/setsize_group1)*sum(u1)-(1/setsize_group2)*sum(u2))
        unfairness = alpha
    return unfairness

def binary_EqOpp(y_actual, y_pred, setsP, classes, pairs):
    #calculates the gap between false negative rates for both groups
    if len(classes)==2: # For Binary Class
        # Assume that the second class of binary class is the advantageous one, so 0 = negative, 1 = positive
        pair = setsP[0] 
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
                if set_group1[idx] == 1: #if in group 1
                    if (y_actual[idx] ==1 and y_pred[idx] ==0): #if acutal class is positive(1), but is classified as negative (0)
                        u1[idx] = 1 #set misclassification vector idx to 1
            # MISCLASSIFCAITON COUTNING VECTOR FOR GROUP G'=2
            for idx in range(0, len(set_group2)): 
                if set_group2[idx] == 1:
                    if (y_actual[idx] ==1 and y_pred[idx] ==0):
                        u2[idx] = 1
        

            alpha = abs((1/setsize_group1)*sum(u1)-(1/setsize_group2)*sum(u2))
            # print('unfairness = ', alpha)
            # quit()
        unfairness = alpha
    return unfairness


def binary_odm(y_actual, y_pred, setsP, classes, pairs):
    #Assume that the second class of binary class is the advantageous one, so 0 = negative, 1 = positive
    pair = setsP[1] 
    set_group1 = pair[0]
    set_group2 = pair[1]
    setsize_group1 = int(sum(set_group1))
    setsize_group2 = int(sum(set_group2))

    print("Lengths:")
    print("y_actual:", len(y_actual))
    print("y_pred:", len(y_pred))
    print("set_group1:", len(set_group1))
    print("set_group2:", len(set_group2))


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