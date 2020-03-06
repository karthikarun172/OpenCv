import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout

os.chdir('/Users/arunkarthik/Downloads/sign_lang')

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


train = pd.read_csv('sign_mnist_train.csv')


labels= train['label'].values

unique_valu = np.array(labels)

train.drop('label',axis=1,inplace=True)

images = train.values
images = np.array([np.reshape(i,(28,28))  for i in images])
images = np.array([i.flatten() for i in images])

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.3,random_state=131)

batch_size = 128
n_classes = 24
epochs = 100

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

model = Sequential()

model.add(Conv2D(64,kernel_size=(3,3),activation='relu' ,input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(n_classes, activation = 'softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = epochs ,batch_size=batch_size)

model.save_weights('/Users/arunkarthik/Downloads/sign_lang/model_weight.h5')
model.save('/Users/arunkarthik/Downloads/sign_lang/gesture_rec.h5')
