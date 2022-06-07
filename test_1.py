import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import unidecode
import csv


def article_tokenize(article):
    article=article.replace("'"," ")
    for i in range(10):
        article=article.replace(str(i)," ")
    for i in range(len(article)):
        lettre=article[i].lower()
        article=article[:i] + lettre + article[i+1:]
    article=unidecode.unidecode(article)
    if type(article)!= str:
        raise Exception("The function takes a string as input data")
    else:
        tokens=word_tokenize(article)
        res=[]
        for x in tokens:
            if len(x)>1:
                res.append(x)
        return res

def langue(article) :
    liste=[0,0,0]
    liste_langue=["french","english","german"]
    for word in article :
        if word in stopwords.words("english") :
            liste[1] += 1
        if word in stopwords.words("french") :
            liste[0] += 1
        if word in stopwords.words("german") :
            liste[2] += 1
    res = max(liste[0],liste[1],liste[2])
    print(liste)
    return liste_langue[liste.index(res)]
  

def remove_stop_words(texte_token ,stop_word):
    res=[]
    for i in texte_token:
        if i not in stop_word:
            res.append(i)
    return res
    

def collection_stemming(segmented_collection,language):
    stemmed_collection=[]
    stemmer = SnowballStemmer(language=str(language))
    for i in segmented_collection:
        stemmed_collection.append(stemmer.stem(i))
    return stemmed_collection


def collection_lemmatize(segmented_collection):
    res=[]
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for i in segmented_collection:
        res.append( stemmer.lemmatize(i))
    return res



Stopwords=set(stopwords.words("english"))
Stopwords=Stopwords.union(set(stopwords.words("french")))
Stopwords=Stopwords.union(set(string.punctuation))
Stopwords=Stopwords.union(set(stopwords.words("german")))


def segmentation (text):
    tokenized_corpus=article_tokenize(text)
    language=langue(tokenized_corpus)
    print(language)
    filtered_collection = remove_stop_words(tokenized_corpus,Stopwords)
    lemmatized_collection=collection_lemmatize(filtered_collection)
    stemmed_collection=collection_stemming(lemmatized_collection,language)
    final=remove_stop_words(stemmed_collection,Stopwords)
    return(final)


texte="Politische Bildung In Der Einwanderungsgesellschaft"
print(segmentation(texte))





def vocabulaire():
    set_final=set()
    file= open (r"./dataset/X_train_update.csv")
    myReader = csv.reader(file)
    x=0
    for row in myReader:
        texte=row[2]
        ensemble=set(segmentation(texte))
        set_final=set_final.union(ensemble)
        ensemble=set(segmentation(texte))
        set_final=set_final.union(ensemble)
        x+=1
        if x==5203:
            return (set_final)

#print(vocabulaire())


