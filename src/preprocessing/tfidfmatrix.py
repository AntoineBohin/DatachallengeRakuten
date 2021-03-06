from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math as m
import pickle
from scipy import sparse

PATH_OF_THE_VOCABULARY_USED_FOR_TRAINING='./dataset/processedWithDescription/vocabulary'

with open (PATH_OF_THE_VOCABULARY_USED_FOR_TRAINING, 'rb') as temp:
        vocabulary_train = pickle.load(temp)

tfidf = TfidfVectorizer(vocabulary=vocabulary_train)

def createTfIdfMatrix_for_evaluation(path_of_the_processed_and_concatenated_X_Y,path_of_the_output_tfidf_matrix,path_of_the_vocabulary=PATH_OF_THE_VOCABULARY_USED_FOR_TRAINING):
    processed_concatenated_df=pd.read_csv(path_of_the_processed_and_concatenated_X_Y, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode', 'TitleAndDescription'],skiprows=[0])
    X_tfidf_evaluation=tfidf.fit_transform(processed_concatenated_df["TitleAndDescription"])
    sparse.save_npz(path_of_the_output_tfidf_matrix, X_tfidf_evaluation)

def processed_and_concatenated_X_Y(path_of_the_processed_X_Y,path_of_the_processed_and_concatenated_X_Y):
    processed_df=pd.read_csv(path_of_the_processed_X_Y, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode'],skiprows=[0])
    processed_df["TitleAndDescription"]=""
    for k in range(len(processed_df)):
        if k%100==0:
            print(k)
        if type(processed_df["Description"][k])==float and m.isnan(processed_df["Description"][k]):
            processed_df["TitleAndDescription"][k]=processed_df["Title"][k]
        else:
            try:    
                processed_df["TitleAndDescription"][k]=processed_df["Title"][k]+processed_df["Description"][k]
            except:
                processed_df["TitleAndDescription"][k]=str(processed_df["Title"][k])+str(processed_df["Description"][k])
    processed_df.to_csv(path_of_the_processed_and_concatenated_X_Y, index=False)

def join_Y_test(Y_test,X_output):
    name_of_the_model=X_output.split('_')[3].split('.')[0]
    y_df=pd.read_csv(Y_test,names=['ProductID', 'RealProductTypeCode'],skiprows=[0])
    x_output=pd.read_csv(X_output,names=['IntegerID', 'CodePrediction'],skiprows=[0])
    x_output["RealProductTypeCode"]=""
    for k in range(len(x_output)):
        x_output["RealProductTypeCode"][k]=y_df['RealProductTypeCode'][k]
    x_output.to_csv("./output/resultat"+name_of_the_model+'.csv', index=False)


join_Y_test('./dataset/baseData/Y_test.csv','./output/prediction_without_labels_logisticregression.csv')

"""
X_tfidf_sample= tfidf.fit_transform(seq)
print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample)
"""
"""
train=pd.read_csv('./dataset/processed_and_concatenated_X_Y_train_with_description.csv', names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode'],skiprows=[0])

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

train.to_csv('processed_and_concatenated_X_Y_train.csv', index=False)
"""
"""
train=pd.read_csv('./dataset/processedWithDescription/processed_and_concatenated_X_Y_train_with_description.csv', names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID', 'ProductTypeCode', 'TitleAndDescription'],skiprows=[0])

X_tfidf_sample=tfidf.fit_transform(train["TitleAndDescription"])



with open('./dataset/processedWithDescription/vocabulary', 'wb') as temp:
    pickle.dump(tfidf.get_feature_names_out(), temp)
with open ('./dataset/processedWithDescription/vocabulary', 'rb') as temp:
    items = pickle.load(temp)
print(items)
print(len(items))
"""
"""
test=pd.DataFrame()
test["test"]=""
test.loc[0]=["hello hello good prediction"]
test.loc[1]=["hello name good good"]
print(test)
X_tfidf_sample=tfidf.fit_transform(test["test"])

print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample.shape)

print("TF-IDF Matrix:")
#print(X_tfidf_sample.todense())
print(tfidf.get_feature_names_out())
print(X_tfidf_sample.todense())
"""
#sparse.save_npz("./dataset/X_consolidated_without_numbers.npz", X_tfidf_sample)