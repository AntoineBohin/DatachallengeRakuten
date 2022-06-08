import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from lightgbm import train
from scipy import sparse


train_csv = pd.read_csv('../../dataset/processed/processed_and_concatenated_X_Y_train.csv', index_col=0)

X_tfidf_sample = sparse.load_npz("../../dataset/X_consolidated_without_numbers.npz")

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_tfidf_sample[0:3000],train_csv["ProductTypeCode"][0:3000])
# predict labels
predictions_SVM = SVM.predict(X_tfidf_sample[60001:84915])
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