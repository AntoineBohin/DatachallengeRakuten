from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd

#from transformers import CamembertTokenizer 
"""
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
res = tokenizer.tokenize("Lolo le crado !")
print(res)
"""

#Cette fonction est utilisée pour nettoyer les textes (titre, descriptif), notamment les bases de HTML ou autres caractères problématiques
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleansp = re.sub('\s+', ' ', cleantext)
    return cleansp

def description_and_title_clean(dataframe):
    dataframe["Description"].apply(cleanhtml)
    dataframe["Title"].apply(cleanhtml)

def process_csv(csv_path_X,csv_path_Y):
    X_train_df=pd.read_csv(csv_path_X, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID'],skiprows=[0])
    Y_train_df=pd.read_csv(csv_path_Y, names=['IntegerID', 'ProductTypeCode'],skiprows=[0])
    train= X_train_df.merge(Y_train_df[['IntegerID', 'ProductTypeCode']], how = 'left', on = 'IntegerID')
    train["Title"]=train["Title"].astype("string")
    train["Description"]=train["Description"].astype("string")  
    
    train["Description"]=train["Description"].apply(cleanhtml)
    train["Title"]=train["Title"].apply(cleanhtml)
    return(train)

train=process_csv('./dataset/X_train_update.csv','./dataset/Y_train_CVw08PX.csv')
print(train)

k=30
print(train["Description"][k])
text=train["Description"][k]


def article_tokenize(text):
    if type(text)!= str:
        raise Exception("The function takes a string as input data")
    else:
        # 1. On extrait les abbreviations
        tokenizer = RegexpTokenizer('[a-zA-Z]\.[a-zA-Z]')#Construction de l'objet tokenizer
        abrevtokens = tokenizer.tokenize(text)#application de la methode tokenize() a l'objet lui permettant d'extraire les abreviations
        # 2. On extrait les mots et les nombres
        tokenizer = RegexpTokenizer('[a-zA-Z]{2,}|\d+\S?\d*')
        wordtokens = tokenizer.tokenize(text)
        return abrevtokens + wordtokens

Stopwords=set(stopwords.words("english"))
Stopwords=Stopwords.union(set(stopwords.words("french")))

#Ajouter méthode td 1 ?
tokenized_corpus=article_tokenize(text)

def remove_stop_words(texte_token ,stop_word):
    res=[]
    for i in texte_token:
        if i not in stop_word:
            res.append(i)
    return res
    

filtered_collection = remove_stop_words(tokenized_corpus,Stopwords)
print(filtered_collection)
