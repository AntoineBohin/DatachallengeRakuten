from pyexpat import model
from lightgbm import train
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import pandas as pd

train_csv = pd.read_csv('processed_and_concatenated_X_Y_train.csv', index_col=0)
X_tfidf_sample = sparse.load_npz("./X_consolidated_without_numbers.npz")


def model_train(Tfidf_matrix_to_train, ProductTypeCodes):
    model= MultinomialNB().fit(Tfidf_matrix_to_train, ProductTypeCodes)
    return(model)


def evaluate_model(Tfidf_matrix_to_evaluate,ProductTypeCodes,model):
    good_predictions=0
    number_of_predictions=0
    for k in range(len(Tfidf_matrix_to_evaluate)):
        predicted=model.predict(Tfidf_matrix_to_evaluate[k])
        number_of_predictions+=1
        if predicted[0]==ProductTypeCodes[k]:
            good_predictions+=1
        print(k,predicted,ProductTypeCodes[k])
    print("Global Accuracy is",good_predictions/number_of_predictions)

if __name__ == '__main__':
    model=model_train(X_tfidf_sample[0:80000],train_csv["ProductTypeCode"][0:80000])
    evaluate_model(X_tfidf_sample[80000:],train_csv["ProductTypeCode"][80000:],model)