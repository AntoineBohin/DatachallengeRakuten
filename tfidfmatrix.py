from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math as m

from scipy import sparse



tfidf = TfidfVectorizer()

"""
"""
train=pd.read_csv('processed_X_Y_train_lolo.csv', names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode'],skiprows=[0])
train["TitleAndDescription"]=""
for k in range(len(train)):
    if k%100==0:
        print(k)
    if type(train["Description"][k])==float and m.isnan(train["Description"][k]):
        train["TitleAndDescription"][k]=train["Title"][k]
    else:
        try:    
            train["TitleAndDescription"][k]=train["Title"][k]+train["Description"][k]
        except:
            train["TitleAndDescription"][k]=str(train["Title"][k])+str(train["Description"][k])
print(train["TitleAndDescription"])
train.to_csv('processed_and_concatenated_X_Y_train_lolo.csv', index=False)

train=pd.read_csv('processed_and_concatenated_X_Y_train_lolo.csv', names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode', 'TitleAndDescription'],skiprows=[0])

X_tfidf_sample=tfidf.fit_transform(train["TitleAndDescription"])


print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample.shape)

print("TF-IDF Matrix:")
#print(X_tfidf_sample.todense())
print(tfidf.get_feature_names())

sparse.save_npz("./X_consolidated_without_numbers_lolo.npz", X_tfidf_sample)
