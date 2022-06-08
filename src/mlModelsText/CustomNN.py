from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from scipy import sparse
import tensorflow
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

train_csv = pd.read_csv('dataset/processed_and_concatenated_X_Y_train_with_description.csv', index_col=0)
X_tfidf_sample = sparse.load_npz("./dataset/X_consolidated_without_numbers_with_description.npz")

DEFAULT_MODEL_PATH='./data/new_model.h5'
BUILD_NEW_NN=False
VOCABULARY_LENGTH=102902


def build_custom_nn():
    model = Sequential()
    model.add(Dense(256, input_dim=VOCABULARY_LENGTH, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(27, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
    return model


def train_model(X_tfidf_sample, ProductTypeCodes,model_path=DEFAULT_MODEL_PATH,build_new_nn=BUILD_NEW_NN):
    if build_new_nn:
        build_custom_nn()
        estimator = KerasClassifier(build_fn=build_custom_nn, epochs=75, batch_size=128)
    else:
        estimator = keras.models.load_model(model_path)

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)                          #We set an EarlyStopping to accelerate the training when the loss is stagnating
    modcheck = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    lb = LabelEncoder()                                                                 #We use a labelEncoder
    y_encoded = lb.fit_transform(ProductTypeCodes)
    processed_y_train = np_utils.to_categorical(y_encoded.ravel())

    j=10000                                                  #As we have to transform the tfidf matrix into an array, there are some problems of memory size allocated as the size of the matrix is ~(84000,102000)
                                                            #So we had to slice the tfidf matrix and to train the model multiple time. If an error occur, you can restart the function with BUILD_NEW_NN=False
    for k in range(3):
        history = estimator.fit(X_tfidf_sample[j-10000:j,:].toarray(), processed_y_train[j-10000:j], epochs=75, batch_size=128, validation_split=0.2, callbacks=[earlystopping, modcheck])
        j+=10000
        print(j)


def evaluate_model_customnn(Tfidf_matrix_to_evaluate,ProductTypeCodes):
    (a,b)=Tfidf_matrix_to_evaluate.shape
    if not b==VOCABULARY_LENGTH:
        print("Your vocabulary size is not the same as the one used by the model !")
    else:
        df_predictions = pd.DataFrame()
        df_predictions["IntegerID"]=""
        df_predictions["CodePredictions"]=""
        df_predictions["RealProductTypeCodes"]=""

        lb = LabelEncoder()
        estimator = keras.models.load_model('./data/new_model_with_description.h5')

        good_predictions=0
        number_of_predictions=0
        for k in range(len(Tfidf_matrix_to_evaluate)):
            y_test = estimator.predict(Tfidf_matrix_to_evaluate[k].toarray())
            y_classes = y_test.argmax(axis=-1)
            y_pred = lb.inverse_transform(y_classes)
            number_of_predictions+=1
            if y_pred[0]==ProductTypeCodes[k]:
                good_predictions+=1
            df_predictions.loc[k]=[k,y_pred[0],ProductTypeCodes[k]]
            print(k,y_pred,ProductTypeCodes[k])
        df_predictions.to_csv('./output/prediction_customNN.csv', index=False)
        print("Global Accuracy is",good_predictions/number_of_predictions)

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tensorflow.SparseTensor(indices, coo.data, coo.shape)


