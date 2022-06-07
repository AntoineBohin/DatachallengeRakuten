import numpy as np 
import math as m 
import random as rd 
import pandas as pd 
import matplotlib.pyplot as plt
from textblob import TextBlob
from charset_normalizer import detect
from sympy import rotations

df_x = pd.read_csv('./dataset/X_train_update.csv')
df_y = pd.read_csv('./dataset/Y_train_CVw08PX.csv', index_col = [0])


## on compte le nombre d'occurences de chaque classe
classes = {}
n = len(df_y)
for i in range(1,n):
    value = df_y.loc[i][0]
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

## Fonction qui classe chaque produit dans une categorie en fonction de son doc_id
def categories(df_x, df_y):
    categories = {}
    n = len(df_y)
    for i in range(n):
        value = df_y.iloc[i]
        product = df_x.iloc[i]
        assert value[0] == product[0]
        if value[0] not in categories.keys():
            categories[value[0]] = [product[3]]
        else: 
            categories[value[0]] = categories[value[0]] + [product[3]]
    return categories

## Renvoie pour chaque categorie les produits qui ont une descritpion ainsi que la liste des produits qui n'en ont pas
def hasDescription(df_x, df_y):
    allDescriptions = {}
    n = len(df_y)
    for i in range(n):
        value = df_y.iloc[i]
        product = df_x.iloc[i]
        description = product['description']
        if type(description) == float and m.isnan(description) :
            if 'No Description' not in allDescriptions.keys():
                allDescriptions['No Description'] = product[0]
            else: 
                allDescriptions['No Description'] = allDescriptions['No Description'] + product[0]
        elif value[0] not in allDescriptions.keys():
            allDescriptions[value[0]] = [product[0]]
        else: 
            allDescriptions[value[0]] = allDescriptions[value[0]] + [product[0]]
    return allDescriptions

## fonction qui pour un article d'entrainement sans description renvoie la description d'un fichier similaire
## (donc de même catégorie) 

allDescriptions = hasDescription(df_x, df_y)
#print(allDescriptions)

def completedDataframe(df_x, df_y):
    df_x2 = df_x.copy(deep=True)
    n = len(df_x)
    for i in range(n):
        value = df_y.loc[i]
        product = df_x.loc[i]
        index = len(allDescriptions[value[0]])
        description = product['description']
        if type(description) == float:
            #print('PAS DE  DESCRIPTION',i)
            newProductID = allDescriptions[value[0]][rd.randint(0,index-1)]
            newProduct = df_x.loc[newProductID]
            newDescription = newProduct['description']
            df_x2.at[i, 'description'] = newDescription
        else: 
            """"la description reste la même car elle existe déjà"""
    return df_x2


new_df = completedDataframe(df_x, df_y)
csv_complet = new_df.to_csv('./X_train_with_description.csv', index = False)

