'''
created by T.E. Röber, March 2024
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import Datasets as DS

def binarize_data(problem, target = 'y', quant=10, save=False):
    '''
    Parameters
    ----------
    problem -- dataset
    target (str) -- name of target variable (default 'y')
    quant (int) -- thresholds to derive from the data (default 10) -- we use deciles
    save (bool) -- save binary files as .csv? (default False)

    Returns
    -------
    df_final -- binarized dataset
    '''

    pname = problem.__name__.upper()
    print(f'---{pname}---')
    df = problem('./datasets/original/') # at this point the data is already one-hot encoded
    df = df.reset_index(drop=True)
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    binary_columns = []
    for column in df.columns:
        # skip target column
        if column == target:
            continue

        unique_values = df[column].unique() # get unique values of columns
        if len(unique_values) <= 2:
            df[column] = df[column].apply(lambda x: 1 if x != unique_values[0] else 0) # make sure they're 0 and 1
            binary_columns.append(column)

    continuous_columns = df.columns.difference(binary_columns+[target])

    # binarize continuous variables
    df_c = pd.DataFrame()
    for column in continuous_columns:
        print(f'binarize column {column}')
        quantiles = np.unique(
            np.quantile(df[column][df[column].notnull()], np.arange(1 / quant, 1, 1 / quant)))
        lowThresh = np.transpose([np.where(df[column] <= x, 1, 0) for x in quantiles])
        highThresh = 1 - lowThresh

        binarized = pd.DataFrame(np.concatenate((lowThresh, highThresh), axis=1))
        binarized.columns = [column + '_' + str(round(x, 2)) + '-' for x in quantiles] + [
            column + '_' + str(round(x, 2)) + '+' for x in quantiles]

        middle_index = len(binarized.columns) // 2  # Calculate the middle index
        df1 = binarized.iloc[:, :middle_index]  # Select columns from the beginning up to the middle index
        df2 = binarized.iloc[:, middle_index:]  # Select columns from the middle index onwards
        for i in range(middle_index):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            binarized = pd.concat([col1, col2], axis=1)
            df_c = pd.concat([df_c, binarized], axis=1)

    # get negations of binary (previously one-hot encoded) data
    df_bin = pd.DataFrame()
    if len(binary_columns) > 0:
        df_bin = df[binary_columns].astype(bool)
        neg = ~df_bin
        neg.columns = [neg.columns[i] + '_neg' for i in range(len(neg.columns))]
        df_bin = pd.concat([df_bin, neg], axis=1)
        df_bin = df_bin.astype(int)

    df_final = pd.concat([df_c, df_bin, df[target]], axis=1)

    print(f'Dataset shape: {df_final.shape}')
    print('---')

    # save df
    if save:
        path = './datasets/binary/'
        if not os.path.exists(path):
            os.makedirs(path)

        df_final.to_csv(f"{path}{pname}_binary.csv", index=False, sep=',')

    return df_final


def save_data_splits(pname, X, y, numSplits=5, randomState=21, data_path='./prepped_data/'):
    kf = KFold(n_splits=numSplits, shuffle=True, random_state=randomState)
    foldnum = 0
    for train_index, val_index in kf.split(X):
        foldnum += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train = pd.DataFrame(X_train)
        train.columns = ['X_' + str(i) for i in range(len(train.columns))]
        train['y'] = y_train
        train.to_csv(f'{data_path}{pname}_train{foldnum}.csv', index=False, sep=';')

        val = pd.DataFrame(X_val)
        val.columns = ['X_' + str(i) for i in range(len(val.columns))]
        val['y'] = y_val
        val.to_csv(f'{data_path}{pname}_val{foldnum}.csv', index=False, sep=';')

    return


# for some methods (e.g. binoct) we need to directly load the training/validation splits as they don't work
# with sklearn's gridsearch cv
# this function splits the data into train and test, and further splits the train data into
def split_data(problem, binary=True, randomState = 0, testSize = 0.2, target = 'y', numSplits=5,
              save_splits=True):
    '''
    Parameters
    ----------
    problem: dataset name as DS.
    binary (bool): use binary data or not? (default = True)
    randomState (int): random state used to split the data (default = 0)
    testSize (float): proportion of test set (default=0.2)
    target (str): name of target column (default = 'y')
    numSplits (int): amount of splits required (for cross validation) (default = 5)
    save_splits (bool): should splits be saved as .csv files? (default = True)


    Returns
    -------
    - Nothing
    - saves the data splits in either ./datasets/train-test-splits/binary/ for binary data, or ./datasets/train-test-splits/original/
    '''

    pname = problem.__name__.upper()
    print(f'---{pname}---')

    if binary:
        df = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
        data_path = './datasets/train-test-splits/binary/'
    else:
        df = problem('./datasets/original/')
        data_path = './datasets/train-test-splits/original/'

    df = df.dropna(subset=[target])
    y = np.array(df[target])
    df = df.drop(target, axis=1)
    X = np.array(df)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=testSize, stratify=y)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save train and test set as .csv files
    train = pd.DataFrame(X_train)
    train.columns = ['X_' + str(i) for i in range(len(train.columns))]
    train['y'] = y_train
    train.to_csv(f"{data_path}{pname}_train_complete.csv", index=False, sep=';')

    test = pd.DataFrame(X_test)
    test.columns = ['X_' + str(i) for i in range(len(test.columns))]
    test['y'] = y_test
    test.to_csv(f"{data_path}{pname}_test.csv", index=False, sep=';')

    if save_splits:
        save_data_splits(pname, X_train, y_train, numSplits=numSplits, randomState=randomState, data_path=data_path)

    return
    # return X_train, X_test, y_train, y_test


# -----
# processing datasets
# -----
numCV: int = 5
testSize = 0.2
randomState = 21

# binary classification
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin,
            DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds,
            DS.default, DS.attrition, DS.student, DS.law, DS.nursery, DS.compas]

for problem in problems:
    # binarize the dataset and save binarized file in ./datasets/binary/
    binarize_data(problem, save=True)

    # for some methods (e.g. binoct) we need to directly load the training/validation splits as they don't work
    # with sklearn's gridsearch cv
    split_data(problem, binary=True, randomState=randomState, testSize=testSize)

    # get dataset info
    pname = problem.__name__.upper()
    df = problem('./datasets/original/')
    df_bin = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
    with open('./datasets/datasets_info.txt', 'a') as f:
        print(f'---{pname}---', file=f)
        print(f'Nr of classes in target: {len(df["y"].unique())}', file=f)
        print(f'Nr of rows: {df.shape[0]}', file=f)
        print(f'Nr of features one-hot-encoded version: {df.shape[1]-1}', file=f)
        print(f'Nr of features binary version: {df_bin.shape[1] - 1}', file=f)
        print('\n', file=f)
