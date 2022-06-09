import numpy as np
import pandas as pd
from PIL import Image
#from tensorflow import keras
#from keras.constraints import maxnorm
#from keras.utils import np_utils

train_csv = pd.read_csv('processed_and_concatenated_X_Y_train.csv', index_col=0)

X_Y_train = [train_csv["ImageID"],train_csv["ProductID"],train_csv["ProductTypeCode"]]
train_images=[]
#print(X_Y_train)
#len(train_csv["ImageID"])
for i in range(1000):
     a = train_csv["ImageID"][i]
     b = train_csv["ProductID"][i]
     c = train_csv["ProductTypeCode"][i]
     image = Image.open( "C:\\Users\\Victor\\Desktop\\DatachallengeRakuten\\images\\image_train\\image_{}_product_{}.jpg".format(a,b) ) 
     im_array = np.array( image )
     train_images.append([im_array,a,c])
print(train_images)
