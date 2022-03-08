# -*- coding: utf-8 -*-
"""
This script has been used to categorize all different disease keywords.
"""

import pandas as pd
from keywords2vec import keywords2vec

# Extract the diseases from the keywords and sort them into the eight categories

path='C:/Users/Klaus/Documents/Python_Scripts/EWBOx2021/Project/training annotation.csv'
df=pd.read_csv(path) # returns a dataframe
# The two columns for the keywords
keywords=df[['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']]
# The eight columns for the categories
allvectors=df[['N','D','G','C','A','H','M','O']]


diseases=[]
for j in range(keywords.shape[0]): # for every line (=pair of eyes)
    for k in range(2): # for each eye individually
        word=keywords.iat[j,k]
        words=word.split(',') # separate the keywords
        for disease in words: # for each single diagnosis
            # remove irrelevant words
            disease=disease.replace('suspected','').replace('suspicious','')
            disease=disease.strip() # remove whitespace at the start and end
            if disease in diseases: # if the diseases is already listed
                pass
            else:
                diseases.append(disease) # if not already listed
# Remove irrelevant keywords from the list
for k in ['normal fundus', 'lens dust', 'low image quality', 'image offset', 'no fundus image',
          'anterior segment image', 'optic disk photographically invisible']:
    diseases.remove(k)

# Create a list of eight lists containing all the keywords corresponding to each category
lists=[[],[],[],[],[],[],[],[]]
for disease in diseases: # For each disease from the list
    vectorlist=[]
    # loop through all the lines of the document; put all the vectors of a patient having
    # this disease (and potentially more) into the list vectorlist
    for j in range(keywords.shape[0]):
        single_keywords=keywords.iat[j,0].split(',')
        single_keywords.extend(keywords.iat[j,1].split(','))
        single_keywords=[s.replace('suspected','').replace('suspicious','').strip() for s in single_keywords]
        if disease in single_keywords:
            vectordf=allvectors.iloc[j]
            vector=vectordf.values.tolist()
            vectorlist.append(vector)
    # Only keep the 1s, which are 1 in every vector in vectorlist to get rid
    # of 1s from other diagnoses
    outvector=[1,1,1,1,1,1,1,1]
    for vector in vectorlist:
        outvector=list(map(lambda x,y: x and y, outvector, vector))
    if sum(outvector)==1: # if the category is unique
        lists[outvector.index(1)].append(disease)
    else: # if the category is not unique
        # Looking at the outputs, all of these should go into the last category "Other diagnoses"
        lists[7].append(disease)
        print(disease)
        print(outvector)
lists[0].append('normal fundus') # The first category is "no disease".


# Test on labels file
for j in range(keywords.shape[0]): # loop through all lines
    vectordf=allvectors.iloc[j]
    vector=vectordf.values.tolist()
    if vector==keywords2vec(list(keywords.iloc[j]), both_sides=True):
        pass
    else:
        print(j)
