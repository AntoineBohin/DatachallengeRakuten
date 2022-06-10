from pyexpat import model
from lightgbm import train
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import pandas as pd
import pickle

train_csv = pd.read_csv('./dataset/processedWithDescription/processed_and_concatenated_X_Y_train_with_description.csv', index_col=0)
#X_tfidf_sample = sparse.load_npz("./dataset/processedWithDescription/X_consolidated_without_numbers_with_description.npz")
X_tfidf_sample = sparse.load_npz("./dataset/X_consolidated_without_numbers.npz")
VOCABULARY_LENGTH=102902
model_path = './models/model_multinomialNB.sav'
 


def model_train(Tfidf_matrix_to_train, ProductTypeCodes):
    model= MultinomialNB().fit(Tfidf_matrix_to_train, ProductTypeCodes)
    pickle.dump(model, open(model_path, 'wb'))

def evaluate_model_nb(Tfidf_matrix_to_evaluate,ProductTypeCodes):
    model=pickle.load(open(model_path, 'rb'))
    df_predictions = pd.DataFrame()
    df_predictions["IntegerID"]=""
    df_predictions["CodePredictions"]=""
    df_predictions["RealProductTypeCodes"]=""
    good_predictions=0
    number_of_predictions=0
    (a,b)=Tfidf_matrix_to_evaluate.shape
    for k in range(a):
        predicted=model.predict(Tfidf_matrix_to_evaluate[k])
        number_of_predictions+=1
        if predicted[0]==ProductTypeCodes.iloc[k]:
            good_predictions+=1
        df_predictions.loc[k]=[k,predicted[0],ProductTypeCodes.iloc[k]]
        #print(k,predicted,ProductTypeCodes.iloc[k])
    df_predictions.to_csv('./output/prediction_MultinomialNB.csv', index=False)

def predict_model_nb(Tfidf_matrix_to_predict,csv_to_predict):
    (a,b)=Tfidf_matrix_to_predict.shape
    if not b==VOCABULARY_LENGTH:
        print("Your vocabulary size is not the same as the one used by the model !")
    else:
        pd.read_csv(csv_to_predict)
        model=pickle.load(open(model_path, 'rb'))

        df_with_information=pd.read_csv(csv_to_predict, names=['IntegerID', 'Title', 'Description', 'ProductID', 'ImageID'],skiprows=[0])

        df_predictions = pd.DataFrame()
        df_predictions["IntegerID"]=""
        df_predictions["CodePredictions"]=""
        (a,b)=Tfidf_matrix_to_predict.shape
        for k in range(a):
            predicted = model.predict(Tfidf_matrix_to_predict[k])
            df_predictions.loc[k]=[df_with_information['IntegerID'][k],predicted[0]]
        df_predictions.to_csv('./output/prediction_without_labels_MultinomialNB.csv', index=False)

if __name__ == '__main__':
    #model_train(X_tfidf_sample[0:80000],train_csv["ProductTypeCode"][0:80000])
    #evaluate_model_nb(X_tfidf_sample[80000:],train_csv["ProductTypeCode"][80000:])
    pass