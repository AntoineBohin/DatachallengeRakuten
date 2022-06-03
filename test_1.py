from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

text='hello, this is a text, bonjour ceci est un texte'


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

