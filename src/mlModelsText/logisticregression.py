import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import pickle

train_csv = pd.read_csv('./dataset/processedWithDescription/processed_and_concatenated_X_Y_train_with_description.csv', index_col=0)
#X_tfidf_sample = sparse.load_npz("./dataset/processedWithDescription/X_consolidated_without_numbers_with_description.npz")
X_tfidf_sample = sparse.load_npz("./dataset/test/TfIdfMatrix_prediction.npz")
model_path='./models/model_logistic_reg.sav'
VOCABULARY_LENGTH=102902

def model_train_logisticregression(Tfidf_matrix_to_train, ProductTypeCodes):
    logclassifier = LogisticRegression(max_iter=100000)
    model=logclassifier.fit(Tfidf_matrix_to_train, ProductTypeCodes)
    pickle.dump(model, open(model_path, 'wb'))

def evaluate_model_logisticregression(Tfidf_matrix_to_evaluate,ProductTypeCodes):
    model=pickle.load(open(model_path, 'rb'))
    df_predictions = pd.DataFrame()
    df_predictions["IntegerID"]=""
    df_predictions["CodePredictions"]=""
    df_predictions["RealProductTypeCodes"]=""

    predictions = model.predict(Tfidf_matrix_to_evaluate)
    good_predictions=0
    number_of_predictions=0
    (a,b)=Tfidf_matrix_to_evaluate.shape
    for k in range(a):
        number_of_predictions+=1
        if predictions[k]==ProductTypeCodes.iloc[k]:
            good_predictions+=1
        df_predictions.loc[k]=[k,predictions[0],ProductTypeCodes.iloc[k]]
    df_predictions.to_csv('./output/prediction_logistic_regression.csv', index=False)

def predict_model_logisticregression(Tfidf_matrix_to_predict,csv_to_predict):
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
        df_predictions.to_csv('./output/prediction_without_labels_logisticregression.csv', index=False)

#model_train_logisticregression(X_tfidf_sample,train_csv["ProductTypeCode"])
predict_model_logisticregression(X_tfidf_sample,'./dataset/baseData/Y_test.csv')
