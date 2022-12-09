"""
@author: sibirbil
"""
import pandas as pd

def loan(wd):
    """
    https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014
    """
    df = pd.read_csv(wd+'loan_data_2007_2014_prepped.csv')
    return df

def banknote(wd): 
    """
    1372 x 5
    2 classes
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    df = pd.read_csv(wd+'data_banknote_authentication.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df
    
def ILPD(wd): 
    """
    583 x 10
    2 classes
    https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    """
    df = pd.read_csv(wd+'ILPD.csv',header = None)
    df.iloc[:,1] = (df.iloc[:,1] == 'Female')*1
    df.iloc[:,-1] = (df.iloc[:,-1] == 2) * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df.dropna(inplace=True)
    return df

def ionosphere(wd): 
    """
    351 x 34
    2 classes
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    df = pd.read_csv(wd+'ionosphere.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 'g')*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def transfusion(wd): 
    """
    748 x 5
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    """
    df = pd.read_csv(wd+'transfusion.csv', header = 0)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def liver(wd): 
    """
    345 x 7
    2 classes
    https://archive.ics.uci.edu/ml/datasets/liver+disorders
    """
    df = pd.read_csv(wd+'bupa.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 2)*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def tictactoe(wd): 
    """
    958 x 9
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    """
    df = pd.read_csv(wd+'tictactoe.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df1 = pd.get_dummies(df.iloc[:,:-1], drop_first=True)
    df1['y'] = (df['y'] == 'positive') *1
    return df1

def wdbc(wd): # Two classes
    """
    569 x 31
    2 classes
    https://datahub.io/machine-learning/wdbc
    """
    df = pd.read_csv(wd+'wdbc.csv', header = None, index_col = 0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = (df['y'] == 'M')*1
    df.drop('y', axis=1, inplace = True)
    df['y'] = y
    return df

def wdbc_original(wd): 
    """
    699 x 10
    2 classes
    https://networkrepository.com/breast-cancer-wisconsin-wdbc.php
    """
    df = pd.read_csv(wd+'wdbc_original.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']  
    y = (df['y'] == 4)*1
    df.drop('y', axis=1, inplace = True)
    df['y'] = y
    return df

def mammography(wd): 
    """
    11183 x 6
    2 classes - Imbalanced
    https://www.openml.org/search?type=data&sort=runs&id=310&status=active
    """
    import pandas as pd
    df = pd.read_csv(wd+'mammography.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def diabetes_pima(wd): 
    """
    768 x 8
    2 classes - Imbalanced
    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    import pandas as pd
    df = pd.read_csv(wd+'diabetes_pima.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def oilspill(wd): 
    """
    937 x 49
    2 classes - Imbalanced
    https://www.kaggle.com/datasets/ashrafkhan94/oil-spill
    """
    import pandas as pd
    df = pd.read_csv(wd+'oilspill.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df = df.drop(df.columns[[0]], axis = 1)
    return df

def phoneme(wd):
    """
    5427 x 6
    2 classes - Imbalanced
    https://datahub.io/machine-learning/phoneme
    """
    import pandas as pd
    df = pd.read_csv(wd+'phoneme.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def seeds(wd): 
    """
    210 x 7
    3 classes
    https://archive.ics.uci.edu/ml/datasets/seeds
    """
    df = pd.read_csv(wd+'seeds.csv', header = None, sep = '\t', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def wine(wd): 
    """
    178 x 13
    3 classes
    https://archive.ics.uci.edu/ml/datasets/wine
    """
    df = pd.read_csv(wd+'wine.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = df['y']
    df.drop('y', axis = 1, inplace = True)
    df['y'] = y
    return df

def glass(wd): 
    """
    214 x 10
    6 classes - Imbalanced
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    df = pd.read_csv(wd+'glass.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1
    return df

def ecoli(wd): 
    """
    336 x 8
    8 classes - Imbalanced
    https://archive.ics.uci.edu/ml/datasets/ecoli
    """
    df = pd.read_csv(wd+'ecoli.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def mushroom(wd):
    """
    8214 x 24
    2 classes
    https://archive.ics.uci.edu/ml/datasets/mushroom
    """
    import pandas as pd
    df = pd.read_csv(wd+'agaricus-lepiota.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    df1 = pd.get_dummies(df.iloc[:,1:], drop_first=True)
    df1['y'] = (df['y'] == 'e') * 1
    return df1

def FICO(wd): 
    """
    9871 x 
    2 classes
    """
    df = pd.read_csv(wd+'FICO_v1.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = df['y']
    df.drop('y', axis = 1, inplace = True)
    df['y'] = y
    return df

def bank_mkt(wd): 
    """
    45211 x 17
    2 classes
    https://archive.ics.uci.edu/ml/datasets/bank+marketing
    """
    
    df = pd.read_csv(wd+'bank_mkt.csv', header = None)
    y = df.iloc[:,-1]
    df.drop(16,inplace=True, axis=1)
    cols_to_encode = [1,2,3,4,6,7,8,10,15]
    df = pd.get_dummies(data = df, columns= cols_to_encode, drop_first=True)
    df['y'] = (y == 'yes')*1
    return df

def hearts(wd): 
    """
    303 x 75
    2 classes
    https://archive.ics.uci.edu/ml/datasets/heart+disease
    """
    df = pd.read_csv(wd+'hearts.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def musk(wd): 
    """
    6589 x 168
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)
    """
    df = pd.read_csv(wd+'musk.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def magic(wd): # Two classes
    """
    19020 x 11
    2 classes
    https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    """
    import pandas as pd
    df = pd.read_csv(wd+'magic.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] = (df['y'] == 'g')*1
    return df


# Fairness Datasets

def student(wd): # Five classes and two groups, sensitive attribute in the first column
    """
    649 x 33
    The sensitive attribute sex is to be put as the first column.
    https://archive.ics.uci.edu/ml/datasets/student+performance
    """
    df = pd.read_csv(wd+'student.csv', header = 1, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    address = df['X_3']
    df = df.drop(columns=['X_3'])
    df.insert(loc=0, column='X_3', value=address)

    return df

def adult(wd): # Two classes
    """
    48842 x 14
    The sensitive attribute sex is to be put as the first column.
    https://archive.ics.uci.edu/ml/datasets/adult
    """
    df = pd.read_csv(wd+'adult.csv', header = None)
    y = df.iloc[:,-1]
    df.drop(14,inplace=True, axis=1)
    cols_to_encode = [1,3,5,6,7,8,13]

    sex = df[9]
    df=df.drop(columns=[9])
    df.insert(loc=1, column='sex', value=sex)
    df = pd.get_dummies(data = df, columns= cols_to_encode, drop_first=False)
    df=df.drop(columns=[0])
    df['y'] = (y == ' >50K')*1
   
    print('Size data set:' , len(df['y']))
 
    return df

def compas(wd):
    """
    2 classes
    6172 x 7 after prepping
    https://github.com/propublica/compas-analysis/
    """
    df = pd.read_csv(wd+'compas.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    return df

def nursery(wd): 
    """
    12960 x 8
    5 classes
    https://archive.ics.uci.edu/ml/datasets/nursery
    """
    df = pd.read_csv(wd+'nursery.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    

    return df

def default(wd):
    """
    30000 x 24
    2 classes
    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    df = pd.read_csv(wd+'default.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    sex = df['X_1']

    df = df.drop(columns=['X_1'])
    df.insert(loc=0, column='X_1', value=sex)
    print('Size data set:' , len(df['y']))
    

    return df

def law(wd): # Five classes
    """
    22387 x 5
    5 classes
    http://www.seaphe.org/databases.php
    """
    df = pd.read_csv(wd+'law.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    return df

def attrition(wd): 
    """
    1469 x 34
    2 classes
    https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    """
    # Dataset IBM HR analytics employee attrition and performance
    df = pd.read_csv(wd+'attrition.csv', header=1, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    string_headers = [2, 4, 7, 14, 16, 20, 21]
    for i in string_headers:
        df['X_'+str(i)] = pd.factorize(df['X_'+str(i)])[0] 

    df['y'] = pd.factorize(df['y'])[0]

    workLifeBalance = df['X_29']
    df = df.drop(columns=['X_29'])
    df.insert(loc=0, column='X_29', value=workLifeBalance)
    return df

def recruitment(wd):
    """
    215 x 12
    8 classes
    https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement
    """
    df = pd.read_csv(wd+'recruitment.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    return df


## Large Classification

def sensorless(wd): 
    """
    58509 x 48
    11 classes
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#Sensorless
    """
    df = pd.read_csv(wd+'sensorless.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1    
    return df

def skinnonskin(wd): 
    """
    245057 x 3
    2 classes
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#skin_nonskin
    """
    df = pd.read_csv(wd+'skinnonskin.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1    
    return df

def covtype(wd): 
    """
    581012 x 54
    7 classes
    https://archive.ics.uci.edu/ml/datasets/covertype
    """
    df = pd.read_csv(wd+'covtype.csv', header = None)
    y = df.iloc[:,-1]-1
    # First ten quantitative columns only  (last one is for y)  
    df = df.iloc[:, 0:11]
    df.columns = ['X_' + str(i) for i in range(10)] + ['y']
    df['y'] = y
    return df

def eicu_mortality(wd):
    df = pd.read_csv(wd+'adult_data_feat_imp.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

