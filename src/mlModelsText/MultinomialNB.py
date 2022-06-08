from pyexpat import model
from lightgbm import train
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import pandas as pd
import pickle

train_csv = pd.read_csv('./dataset/processedWithDescription/processed_and_concatenated_X_Y_train_with_description.csv', index_col=0)
X_tfidf_sample = sparse.load_npz("./X_consolidated_without_numbers.npz")

model_path = './models/model_multinomialNB.sav'
 


def model_train(Tfidf_matrix_to_train, ProductTypeCodes):
    model= MultinomialNB().fit(Tfidf_matrix_to_train, ProductTypeCodes)
    pickle.dump(model, open(model_path, 'wb'))

def evaluate_model_nb(Tfidf_matrix_to_evaluate,ProductTypeCodes):
    pickle.load(open(model_path, 'rb'))
    df_predictions = pd.DataFrame()
    df_predictions["IntegerID"]=""
    df_predictions["CodePredictions"]=""
    df_predictions["RealProductTypeCodes"]=""
    good_predictions=0
    number_of_predictions=0
    for k in range(len(Tfidf_matrix_to_evaluate)):
        predicted=model.predict(Tfidf_matrix_to_evaluate[k])
        number_of_predictions+=1
        if predicted[0]==ProductTypeCodes[k]:
            good_predictions+=1
        df_predictions.loc[k]=[k,predicted[0],ProductTypeCodes[k]]
        #print(k,predicted,ProductTypeCodes[k])
    df_predictions.to_csv('./output/prediction_MultinomialNB.csv', index=False)
    print("Global Accuracy is",good_predictions/number_of_predictions)

if __name__ == '__main__':
    model_train(X_tfidf_sample[0:80000],train_csv["ProductTypeCode"][0:80000])
    evaluate_model_nb(X_tfidf_sample[80000:],train_csv["ProductTypeCode"][80000:])