from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import unidecode
import string


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

def to_string(classe):
    return(str(classe))

def description_and_title_clean(dataframe):
    dataframe["Description"].apply(cleanhtml)
    dataframe["Title"].apply(cleanhtml)

def process_csv(csv_path_X,csv_path_Y):
    X_train_df=pd.read_csv(csv_path_X, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID'],skiprows=[0])
    Y_train_df=pd.read_csv(csv_path_Y, names=['IntegerID', 'ProductTypeCode'],skiprows=[0])
    train= X_train_df.merge(Y_train_df[['IntegerID', 'ProductTypeCode']], how = 'left', on = 'IntegerID')
    train["Title"]=train["Title"].astype("string")
    train["Description"]=train["Description"].astype("string")  
    return(train)

def article_tokenize(article):
    article=article.replace("'"," ")
    for i in range(10):
        article=article.replace(str(i)," ")
    article=unidecode.unidecode(article)
    if type(article)!= str:
        raise Exception("The function takes a string as input data")
    else:
        tokens=word_tokenize(article)
        return tokens

Stopwords=set(stopwords.words("english"))
Stopwords=Stopwords.union(set(stopwords.words("french")))
Stopwords=Stopwords.union(set(string.punctuation))

#Ajouter méthode td 1 ?

def remove_stop_words(texte_token ,stop_word):
    res=[]
    for i in texte_token:
        if i not in stop_word:
            res.append(i)
    return res

def collection_stemming(segmented_collection):
    stemmed_collection=[]
    stemmer = SnowballStemmer(language='french')
    #stemmer = PorterStemmer ()
    for i in segmented_collection:
        stemmed_collection.append(stemmer.stem(i))
    return stemmed_collection

def collection_lemmatize(segmented_collection):
    res=[]
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for i in segmented_collection:
        res.append( stemmer.lemmatize(i))
    return res

def segmentation(text):
    tokenized_corpus=article_tokenize(text)
    filtered_collection = remove_stop_words(tokenized_corpus,Stopwords)
    lemmatized_collection=collection_lemmatize(filtered_collection)
    stemmed_collection=collection_stemming(lemmatized_collection)
    final=remove_stop_words(stemmed_collection,Stopwords)
    return(final)


def concatenate_text(list):
    string=""
    for k in range(len(list)):
        string=string+list[k]+" "
    return(string)


def text_treatment_of_the_csv(dataframe):
    for k in range(len(dataframe)):
        description=dataframe["Description"][k]
        title=dataframe["Title"][k]
        try:    
            dataframe["Title"][k]=cleanhtml(dataframe["Title"][k])
        except:
            """do nothing"""
        try:
            dataframe["Description"][k]=cleanhtml(dataframe["Description"][k])
        except:
            """do nothing"""
        description=segmentation(str(description))
        description=concatenate_text(description)
        title=segmentation(str(title))
        title=concatenate_text(title)
        dataframe["Description"][k]=description
        dataframe["Title"][k]=title
    return(dataframe)

def csv_pipeline(csv_path_X,csv_path_Y):
    train=process_csv(csv_path_X,csv_path_Y)
    train=text_treatment_of_the_csv(train)
    print(train)
    return(train)


train=csv_pipeline('./dataset/X_train_update.csv','./dataset/Y_train_CVw08PX.csv')

#filtered_collection = remove_stop_words(tokenized_corpus,Stopwords)
#print(filtered_collection)

train.to_csv('processed_X_Y_train.csv', index=False)


tfidf = TfidfVectorizer()

"""
X_tfidf_sample= tfidf.fit_transform(seq)
print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample)
"""