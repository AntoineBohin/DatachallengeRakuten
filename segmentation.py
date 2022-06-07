import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import unidecode



def article_tokenize(article):
    article=article.replace("'"," ")
    article=unidecode.unidecode(article)
    if type(article)!= str:
        raise Exception("The function takes a string as input data")
    else:
        tokens=word_tokenize(article)
        return tokens


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



Stopwords=set(stopwords.words("english"))
Stopwords=Stopwords.union(set(stopwords.words("french")))
Stopwords=Stopwords.union(set(string.punctuation))


def segmentation (text):
    tokenized_corpus=article_tokenize(text)
    filtered_collection = remove_stop_words(tokenized_corpus,Stopwords)
    lemmatized_collection=collection_lemmatize(filtered_collection)
    stemmed_collection=collection_stemming(lemmatized_collection)
    final=remove_stop_words(stemmed_collection,Stopwords)
    return(final)


