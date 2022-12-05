"""
@author: sibirbil
"""
import pandas as pd

def loan(wd):
    df = pd.read_csv(wd+'loan_data_2007_2014_prepped.csv')
    return df

def banknote(wd): # Two classes
    """
    Attribute Information:
    1. variance of Wavelet Transformed image (continuous)
    2. skewness of Wavelet Transformed image (continuous)
    3. curtosis of Wavelet Transformed image (continuous)
    4. entropy of image (continuous)
    5. class (integer)
    """
    df = pd.read_csv(wd+'data_banknote_authentication.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df
    
def ILPD(wd): # Two classes
    """
    Attribute Information:
    1. Age Age of the patient
    2. Gender Gender of the patient
    3. TB Total Bilirubin
    4. DB Direct Bilirubin
    5. Alkphos Alkaline Phosphotase
    6. Sgpt Alamine Aminotransferase
    7. Sgot Aspartate Aminotransferase
    8. TP Total Protiens
    9. ALB Albumin
    10. A/G Ratio Albumin and Globulin Ratio
    11. Selector field used to split the data into 
        two sets (labeled by the experts)
    """
    df = pd.read_csv(wd+'ILPD.csv',header = None)
    df.iloc[:,1] = (df.iloc[:,1] == 'Female')*1
    df.iloc[:,-1] = (df.iloc[:,-1] == 2) * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df.dropna(inplace=True)
    return df

def ionosphere(wd): # Two classes
    """
    Attribute Information:

    -- All 34 are continuous
    -- The 35th attribute is either "good" or "bad" 
       according to the definition summarized above. 
       This is a binary classification task.
    """
    df = pd.read_csv(wd+'ionosphere.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 'g')*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def transfusion(wd): # Two classes
    """
    Attribute Information:
    Given is the variable name, variable type, the measurement unit and a 
    brief description. The "Blood Transfusion Service Center" is a 
    classification problem. The order of this listing corresponds to the 
    order of numerals along the rows of the database.

    R (Recency - months since last donation),
    F (Frequency - total number of donation),
    M (Monetary - total blood donated in c.c.),
    T (Time - months since first donation), and
    a binary variable representing whether he/she donated blood in March 2007 
    (1 stand for donating blood; 0 stands for not donating blood).
    """
    df = pd.read_csv(wd+'transfusion.csv', header = 0)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def liver(wd): # Two classes
    """
    Attribute information:
    1. mcv	mean corpuscular volume
    2. alkphos	alkaline phosphotase
    3. sgpt	alamine aminotransferase
    4. sgot 	aspartate aminotransferase
    5. gammagt	gamma-glutamyl transpeptidase
    6. drinks	number of half-pint equivalents of 
       alcoholic beverages drunk per day
    7. selector  field used to split data into two set
   """
    df = pd.read_csv(wd+'bupa.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 2)*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def tictactoe(wd): # Two classes
    """
    Attribute Information:
    1. top-left-square: {x,o,b}
    2. top-middle-square: {x,o,b}
    3. top-right-square: {x,o,b}
    4. middle-left-square: {x,o,b}
    5. middle-middle-square: {x,o,b}
    6. middle-right-square: {x,o,b}
    7. bottom-left-square: {x,o,b}
    8. bottom-middle-square: {x,o,b}
    9. bottom-right-square: {x,o,b}
    10. Class: {positive,negative}
    """
    df = pd.read_csv(wd+'tictactoe.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df1 = pd.get_dummies(df.iloc[:,:-1], drop_first=True)
    df1['y'] = (df['y'] == 'positive') *1
    return df1

def wdbc(wd): # Two classes
    """
    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32)
    Ten real-valued features are computed for each cell nucleus:
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry 
        j) fractal dimension ("coastline approximation" - 1)
    """
    df = pd.read_csv(wd+'wdbc.csv', header = None, index_col = 0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = (df['y'] == 'M')*1
    df.drop('y', axis=1, inplace = True)
    df['y'] = y
    return df

def wdbc_original(wd): # Two classes

    df = pd.read_csv(wd+'wdbc_original.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']  
    y = (df['y'] == 4)*1
    df.drop('y', axis=1, inplace = True)
    df['y'] = y
    return df

def mammography(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    7. Class (-1 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'mammography.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def diabetes_pima(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'diabetes_pima.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def oilspill(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    x. Class (0 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'oilspill.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df = df.drop(df.columns[[0]], axis = 1)
    return df

def phoneme(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    Five different attributes were chosen to
    characterize each vowel: they are the amplitudes of the five first
    harmonics AHi, normalised by the total energy Ene (integrated on all the
    frequencies): AHi/Ene. Each harmonic is signed: positive when it
    corresponds to a local maximum of the spectrum and negative otherwise.
    6. Class (0 and 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'phoneme.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def seeds(wd): # Three classes
    """
    Attribute Information:

    To construct the data, seven geometric parameters of wheat kernels were measured:
    1. area A,
    2. perimeter P,
    3. compactness C = 4*pi*A/P^2,
    4. length of kernel,
    5. width of kernel,
    6. asymmetry coefficient
    7. length of kernel groove.
    All of these parameters were real-valued continuous.
    """
    df = pd.read_csv(wd+'seeds.csv', header = None, sep = '\t', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def wine(wd): # Three classes
    """
    The attributes are donated by Riccardo Leardi (riclea@anchem.unige.it)
 	 1) Alcohol
 	 2) Malic acid
 	 3) Ash
	 4) Alcalinity of ash  
 	 5) Magnesium
	 6) Total phenols
 	 7) Flavanoids
 	 8) Nonflavanoid phenols
 	 9) Proanthocyanins
	10) Color intensity
 	11) Hue
 	12) OD280/OD315 of diluted wines
 	13) Proline            
    Number of Instances
    class 1 59
	class 2 71
	class 3 48
    """
    df = pd.read_csv(wd+'wine.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = df['y']
    df.drop('y', axis = 1, inplace = True)
    df['y'] = y
    return df

def glass(wd): # Six classes - Imbalanced
    """
    Attribute Information:
    RI: refractive index
    Na: Sodium
    Mg: Magnesium
    Al: Aluminum
    Si: Silicon
    K: Potassium
    Ca: Calcium
    Ba: Barium
    Fe: Iron

    Class 1: building windows (float processed)
    Class 2: building windows (non-float processed)
    Class 3: vehicle windows (float processed)
    Class 4: containers
    Class 5: tableware
    Class 6: headlamps

    """
    df = pd.read_csv(wd+'glass.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1
    return df

def ecoli(wd): # Eight classes - Imbalanced
    """
    Attribute Information:
    0: McGeoch’s method for signal sequence recognition
    1: von Heijne’s method for signal sequence recognition
    2: von Heijne’s Signal Peptidase II consensus sequence score
    3: Presence of charge on N-terminus of predicted lipoproteins
    4: Score of discriminant analysis of the amino acid content 
       of outer membrane and periplasmic proteins.
    5: score of the ALOM membrane-spanning region prediction program
    6: score of ALOM program after excluding putative cleavable 
       signal regions from the sequence.
    
    Eight Classes:
    0: cytoplasm
    1: inner membrane without signal sequence
    2: inner membrane lipoprotein
    3: inner membrane, cleavable signal sequence
    4: inner membrane, non cleavable signal sequence
    5: outer membrane
    6: outer membrane lipoprotein
    7: periplasm
    """
    df = pd.read_csv(wd+'ecoli.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def mushroom(wd): # Two classes
    """
    6. Number of Attributes: 22 (all nominally valued)

    7. Attribute Information: (classes: edible=e, poisonous=p)
         1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                      knobbed=k,sunken=s
         2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
         3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                      pink=p,purple=u,red=e,white=w,yellow=y
         4. bruises?:                 bruises=t,no=f
         5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                      musty=m,none=n,pungent=p,spicy=s
         6. gill-attachment:          attached=a,descending=d,free=f,notched=n
         7. gill-spacing:             close=c,crowded=w,distant=d
         8. gill-size:                broad=b,narrow=n
         9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                      green=r,orange=o,pink=p,purple=u,red=e,
                                      white=w,yellow=y
        10. stalk-shape:              enlarging=e,tapering=t
        11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                      rhizomorphs=z,rooted=r,missing=?
        12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
        13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
        14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                      pink=p,red=e,white=w,yellow=y
        15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                      pink=p,red=e,white=w,yellow=y
        16. veil-type:                partial=p,universal=u
        17. veil-color:               brown=n,orange=o,white=w,yellow=y
        18. ring-number:              none=n,one=o,two=t
        19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                      none=n,pendant=p,sheathing=s,zone=z
        20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                      orange=o,purple=u,white=w,yellow=y
        21. population:               abundant=a,clustered=c,numerous=n,
                                      scattered=s,several=v,solitary=y
        22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                      urban=u,waste=w,woods=d

    8. Missing Attribute Values: 2480 of them (denoted by "?"), all for
       attribute #11.
       """
    import pandas as pd
    df = pd.read_csv(wd+'agaricus-lepiota.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    df1 = pd.get_dummies(df.iloc[:,1:], drop_first=True)
    df1['y'] = (df['y'] == 'e') * 1
    return df1

def FICO(wd): # Two classes
    
    import pandas as pd
    df = pd.read_csv(wd+'FICO_v1.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = df['y']
    df.drop('y', axis = 1, inplace = True)
    df['y'] = y
    return df

def bank_mkt(wd): # Two classes
    
    import pandas as pd
    df = pd.read_csv(wd+'bank_mkt.csv', header = None)
    y = df.iloc[:,-1]
    df.drop(16,inplace=True, axis=1)
    cols_to_encode = [1,2,3,4,6,7,8,10,15]
    df = pd.get_dummies(data = df, columns= cols_to_encode, drop_first=True)
    df['y'] = (y == 'yes')*1
    return df

def hearts(wd): # Two classes
    
    import pandas as pd
    df = pd.read_csv(wd+'hearts.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def musk(wd): # Two classes
    
    import pandas as pd
    df = pd.read_csv(wd+'musk.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def magic(wd): # Two classes
    """
    7. Attribute information:
        1.  fLength:  continuous  # major axis of ellipse [mm]
        2.  fWidth:   continuous  # minor axis of ellipse [mm] 
        3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
        4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
        5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
        6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
        7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
        8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
        9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
       10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
       11.  class:    g,h         # gamma (signal), hadron (background)
    """
    import pandas as pd
    df = pd.read_csv(wd+'magic.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] = (df['y'] == 'g')*1
    return df

# Fairness
def student(wd): # Five classes and two groups, sensitive attribute in the first column
    """
    Sensitive attribute: gender
    Groups: male, female

    Class 1: grade A
    Class 2: grade B
    Class 3: grade C
    Class 4: grade D
    Class 5: grade E
    """
    df = pd.read_csv(wd+'student.csv', header = 1, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    address = df['X_3']
    df = df.drop(columns=['X_3'])
    df.insert(loc=0, column='X_3', value=address)

    return df

def adult(wd): # Two classes
    import pandas as pd
    df = pd.read_csv(wd+'adult.csv', header = None)
    y = df.iloc[:,-1]
    df.drop(14,inplace=True, axis=1)

    cols_to_encode = [1,3,5,6,7,8,13]
    # Put sensitive attribute first
    sex = df[9]
    df=df.drop(columns=[9])
    df.insert(loc=1, column='sex', value=sex)
    df = pd.get_dummies(data = df, columns= cols_to_encode, drop_first=False)
    df=df.drop(columns=[0])
    df['y'] = (y == ' >50K')*1
   

    print('Size data set:' , len(df['y']))
 
    return df

def compas(wd):
    df = pd.read_csv(wd+'compas.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    return df

def nursery(wd): # Five classes
    df = pd.read_csv(wd+'nursery.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    

    return df

def default(wd): # Five classes
    df = pd.read_csv(wd+'default.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    sex = df['X_1']

    df = df.drop(columns=['X_1'])
    df.insert(loc=0, column='X_1', value=sex)
    print('Size data set:' , len(df['y']))
    

    return df

def law(wd): # Five classes
    df = pd.read_csv(wd+'law.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    return df

def attrition(wd): 
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
    df = pd.read_csv(wd+'recruitment.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))

    return df

## Large Classification

def sensorless(wd): # 11 classes
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#Sensorless
    df = pd.read_csv(wd+'sensorless.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1    
    return df

def skinnonskin(wd): # 2 classes
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#skin_nonskin
    df = pd.read_csv(wd+'skinnonskin.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1    
    return df

def covtype(wd): # Seven classes
# https://archive.ics.uci.edu/ml/datasets/covertype
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

# def adult(wd): # Two classes
#     """
#     age: continuous.
#     workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#     fnlwgt: continuous.
#     education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#     education-num: continuous.
#     marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#     occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#     relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#     race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#     sex: Female, Male.
#     capital-gain: continuous.
#     capital-loss: continuous.
#     hours-per-week: continuous.
#     native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#     """
#     import pandas as pd
#     df = pd.read_csv(wd+'adult.csv', header = None)
#     y = df.iloc[:,-1]
#     df.drop(14,inplace=True, axis=1)

#     cols_to_encode = [1,3,5,6,7,8,13]
#     #put sensitive attribute first
#     sex = df[9]
#     df=df.drop(columns=[9])
#     df.insert(loc=1, column='sex', value=sex)
#     df = pd.get_dummies(data = df, columns= cols_to_encode, drop_first=False)
#     df=df.drop(columns=[0])
#     df['y'] = (y == ' >50K')*1

#     print('Size data set:' , len(df['y']))
 
#     return df