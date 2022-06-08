import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

Y = pd.read_csv('dataset/processed_and_concatenated_X_Y_train_lolo.csv', index_col=0)
X = sparse.load_npz("./dataset/X_consolidated_without_numbers_lolo.npz")
 
(X_train, Y_train) = (X[:80000],Y["ProductTypeCode"][:80000])
(X_test, Y_test)= (X[80000:],Y["ProductTypeCode"][80000:])

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
score = classifier.score(X_test, Y_test)
print("Accuracy:", score)



