import numpy as np 
import math as m 
import pandas as pd 
import matplotlib.pyplot as plt
from textblob import TextBlob
from charset_normalizer import detect
from sympy import rotations

df_x = pd.read_csv('./dataset/X_train_update.csv')
df_y = pd.read_csv('./dataset/Y_train_CVw08PX.csv')

## on compte le nombre d'occurences de chaque classe
classes = {}
n = len(df_y)
for i in range(1,n):
    value = df_y.iloc[i][1]
    if value not in classes.keys():
        classes[value] = 1
    else :
        classes[value] += 1

def somme(dict): 
    somme = 0
    for key in dict.keys():
        somme += dict[key]
    return somme 

def frequency(dict):
    new_dict = {}
    n = somme(dict)
    for key in dict.keys():
        new_dict[key] = dict[key]/n*100
    return new_dict

frequences = frequency(classes)

list = [frequences[key] for key in sorted(frequences.keys())]

plt.bar(range(len(list)), list)

plt.title("fréquences d'apparition dans chaque catégorie")
plt.ylabel("frequence (%)")
plt.xticks(range(len(frequences.keys())), sorted(frequences.keys()),rotation='vertical')
#plt.show()

## on compte le nombre de description vides parmi l'ensemble des descriptions 
def pagesVides():
    pagesVides = {}
    pagesVides['vide'] = 0
    pagesVides['non_vide'] = 0
    for i in range(len(df_x)):
        description = df_x.iloc[i]['description']
        if type(description) == float and m.isnan(description) :
            pagesVides['vide'] +=1
        else: 
            pagesVides['non_vide'] += 1
    n = somme(pagesVides)
    pagesVides['vide'] /= n
    pagesVides['non_vide'] /= n 
    return pagesVides

## on compte le nombre de description en langue autre que le français 
def langues():
    langues = {}
    for i in range(len(df_x)):
        description = df_x.iloc[i]['description']
        if not type(description) == float: 
            doc = TextBlob(description)
            detect_language = doc.detect_language()
            if not detect_language in langues.keys() :
                langues[detect_language] = 1
            else: 
                langues[detect_language] += 1
    return langues

print(langues())