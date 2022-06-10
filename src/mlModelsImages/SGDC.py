import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.datasets import mnist
import os
import tensorflow as tf
from PIL import Image
import sys
import pickle
from sklearn.linear_model import SGDClassifier
 
classe=[10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905]
classetest=[i for i in range(27)]
def folder_to_numpy(foldertrain,taille,début):
    X,y=[],[]
    k=0
    etat='train'
    for filename in os.listdir(foldertrain)[début:]:
        print(k)
        image=tf.keras.utils.img_to_array(Image.open(foldertrain+ filename).resize((250,250)), data_format=None, dtype=None)
        image=rgb2gray(image)
        image=image.reshape(250*250)
        prdt_type=get_producttype_fromid(filename)
        X.append(image)
        y.append(classe.index(prdt_type))
        k+=1
        if k==taille:
            #np.save('x_'+str(début),X)
            #np.save('y_'+str(début),y)
            X=np.array(X)
            y=np.array(y)
            return (X,y)
 
def get_producttype_fromid(imagename):
    list=imagename.split('_')
    product_type=list[4].split('.')[0]
    return(int(product_type))
 
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255
 
"""
clf = SGDClassifier(max_iter=1000000, tol=0.01)
foldertrain = "./images/image_train/"
taille=15000
X,y=folder_to_numpy(foldertrain,taille,0)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,train_size=12000,test_size=3000)
clf.fit(xtrain,ytrain)
score = clf.score(xtest, ytest)
print("Accuracy:", score)
 
"""
#### On peut aussi grace a cette méthode séparer les données. Cela permet de traiter plus de données et d'obtenir de meilleurs résultats
 
taille=10000
foldertrain = "./images/image_train/"
xtest,ytest=folder_to_numpy(foldertrain,taille,0)
clf = SGDClassifier(max_iter=1000000, tol=0.01)
for i in range(12):
    print(i)
    xtrain,ytrain=folder_to_numpy(foldertrain,taille,10000+5000*i)
    clf.partial_fit(xtrain,ytrain,classes=classetest)
score = clf.score(xtest, ytest)
print("Accuracy:", score)