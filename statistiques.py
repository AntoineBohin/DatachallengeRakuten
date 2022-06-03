import numpy as np 
import math as m 
import pandas as pd 
import matplotlib.pyplot as plt
import spacy

from spacy_langdetect import LanguageDetector
from sympy import rotations

df_x = pd.read_csv('./dataset/X_train_update.csv')
df_y = pd.read_csv('./dataset/Y_train_CVw08PX.csv')

print(m.isnan(df_x.iloc[5]['description']))

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
        if m.isnan(description) :
            print(description)
            pagesVides['vide'] +=1
        else: 
            pagesVides['non_vide'] += 1
    return pagesVides

print(pagesVides())

## on compte le nombre de description en langue autre que le français 
def langues():
    nlp = spacy.load('xx_ent_wiki_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    langues = {}
    for i in range(len(df_x)):
        description = df_x.iloc[i]['description']
        doc = nlp(description)
        detect_language = doc._.language
        if not detect_language in langues.keys() :
            langues[detect_language] = 1
        else: 
            langues[detect_language] += 1
    return langues
