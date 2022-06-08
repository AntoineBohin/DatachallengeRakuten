from lightgbm import train
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import pandas as pd

train_csv = pd.read_csv('./dataset/processed_and_concatenated_X_Y_train.csv', index_col=0)

X_tfidf_sample = sparse.load_npz("./dataset/X_consolidated_without_numbers.npz")



clf= MultinomialNB().fit(X_tfidf_sample[0:80000],train_csv["ProductTypeCode"][0:80000])


good_predictions=0
number_of_predictions=0
for k in range(80001,84915):
    predicted=clf.predict(X_tfidf_sample[k])
    number_of_predictions+=1
    if predicted[0]==train_csv["ProductTypeCode"][k]:
        good_predictions+=1
    print(k,predicted,train_csv["ProductTypeCode"][k])

print("Global Accuracy is",good_predictions/number_of_predictions)