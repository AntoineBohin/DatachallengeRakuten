import detectlanguage
from nltk.corpus import stopwords

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

Stopwords=set(stopwords.words("english"))
Stopwords=Stopwords.union(set(stopwords.words("french")))
Stopwords=Stopwords.union(set(stopwords.words("german")))

article="bonjour je m'appelle nicolas"
print(article)
detectlanguage.configuration.api_key = '0a2071cc13cbb9d7359d0c44eed1c323'
print(detectlanguage.simple_detect(article))
print(langue(article))

