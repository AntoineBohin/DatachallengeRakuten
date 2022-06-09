import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from lightgbm import train
from scipy import sparse
import pickle

train_csv = pd.read_csv('./dataset/processedWithDescription/processed_and_concatenated_X_Y_train_with_description.csv', index_col=0)
X_tfidf_sample = sparse.load_npz("./dataset/processedWithDescription/X_consolidated_without_numbers_with_description.npz")
model_path='./models/model svm 88%.sav'
VOCABULARY_LENGTH=102902


def model_train_svm(Tfidf_matrix_to_train, ProductTypeCodes):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model= SVM().fit(Tfidf_matrix_to_train, ProductTypeCodes)
    pickle.dump(model, open(model_path, 'wb'))

def evaluate_model_svm(Tfidf_matrix_to_evaluate,ProductTypeCodes):
    SVM=pickle.load(open(model_path, 'rb'))
    df_predictions = pd.DataFrame()
    df_predictions["IntegerID"]=""
    df_predictions["CodePredictions"]=""
    df_predictions["RealProductTypeCodes"]=""

    predictions_SVM = SVM.predict(Tfidf_matrix_to_evaluate)
    good_predictions=0
    number_of_predictions=0
    (a,b)=Tfidf_matrix_to_evaluate.shape
    for k in range(a):
        number_of_predictions+=1
        if predictions_SVM[k]==ProductTypeCodes.iloc[k]:
            good_predictions+=1
        df_predictions.loc[k]=[k,predictions_SVM[0],ProductTypeCodes.iloc[k]]
    df_predictions.to_csv('./output/prediction_MultinomialNB.csv', index=False)

def predict_model_svm(Tfidf_matrix_to_predict,csv_to_predict):
    (a,b)=Tfidf_matrix_to_predict.shape
    if not b==VOCABULARY_LENGTH:
        print("Your vocabulary size is not the same as the one used by the model !")
    else:
        SVM=pickle.load(open(model_path, 'rb'))
        df_with_information=pd.read_csv(csv_to_predict, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID'],skiprows=[0])

        df_predictions = pd.DataFrame()
        df_predictions["IntegerID"]=""
        df_predictions["CodePredictions"]=""

        predictions_SVM = SVM.predict(Tfidf_matrix_to_predict)
        (a,b)=Tfidf_matrix_to_predict.shape
        for k in range(a):
            df_predictions.loc[k]=[df_with_information['IntegerID'][k],predictions_SVM[k]]
        df_predictions.to_csv('./output/prediction_without_labels_svm.csv', index=False)

predict_model_svm(X_tfidf_sample[:100],'./dataset/processedWithDescription/processed_X_Y_train_with_description.csv')


"""
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_tfidf_sample[0:300],train_csv["ProductTypeCode"][0:300])
# predict labels
predictions_SVM = SVM.predict(X_tfidf_sample[60001:84915])
print(predictions_SVM[1])
# get the accuracy
#print("Accuracy: ",accuracy_score(predictions_SVM, train_csv["ProductTypeCode"][80001:84915])*100)
import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(SVM, open(filename, 'wb'))
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
#from joblib import dump, load

#dump(SVM, 'model_svm_saved.joblib')
"""