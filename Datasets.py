#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21 April 2021

@author: sibirbil
"""
import pandas as pd

def banknote(wd): # Two classes
    """
    Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
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
    
def seeds(wd): # Three classes
    """
    Source: https://archive.ics.uci.edu/ml/datasets/seeds
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

def glass(wd): # Six classes - Imbalanced
    """
    Source: https://archive.ics.uci.edu/ml/datasets/Glass+Identification
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
    Source: https://archive.ics.uci.edu/ml/datasets/Ecoli
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
