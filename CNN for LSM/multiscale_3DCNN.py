# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from tensorflow.keras.layers import BatchNormalization,AveragePooling2D,concatenate
from tensorflow.keras.layers import ZeroPadding2D,add
from tensorflow.keras.layers import Dropout, Activation
from tensorflow import keras
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers, regularizers # 优化器，正则化项
from tensorflow.keras.optimizers import SGD, Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''focal loss function'''
def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*' * 20, 'alpha={}, gamma={}'.format(alpha, gamma))

    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))

        loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_sum(loss)

    return focal_loss_calc
'''
Step1 Read data from csv
df1 and df2 are the datarrame variables of the read csv.
'''

df1 = pd.read_csv(r"E:\CNN-res\new\i341_j368.csv")
df2 = pd.read_csv(r"E:\CNN-res\new\i580_j316.csv")

'''
step2 Handling Missing Values 
data1 and data2 are the tags that complement the missing values.
'''
data1 = df1.fillna(0)
data2 = df2.fillna(0)

'''
step3 Normalisation of features
Normalisation of landslide impact factors by features
'''
minmax = preprocessing.MinMaxScaler()
data1 = minmax.fit_transform(data1[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])
data2 = minmax.fit_transform(data2[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])

print(data1)
print(data2)

'''
step4 
x1_data1 get into 3D format number of samples * number of features * channels
y1_data1 is an array of real labels
'''
x1_data1 = np.expand_dims(data1.astype(float), axis=2)
y1_data1 = df1.values[:, -2]
print(y1_data1)

x1_data2 = np.expand_dims(data2.astype(float), axis=2)
y1_data2 = df2.values[:, -2]
x2_data1 = np.expand_dims(data1.astype(float), axis=2)
y2_data1 = df1.values[:, -2]
'''
step5
Multi-scale sampling strategy
'''
print(1)
I = 0
J = 0
K = 0
i = 0
j = 0
k = 0
X_data = []
Y_data = []
while J < 330:
    i = 0
    while i < 357:
        j = 0
        while j < 12:
            X_data.append(x1_data1[i + 357 * J + j * 357:(i + 357 * J + j * 357) + 12, :])
            j = j + 1
        i = i + 1
    J = J + 1
I = 0
J = 0
K = 0
i = 0
j = 0
k = 0
while J < 330:
    i = 0
    while i < 357:
        Y_data.append(y1_data1[i + 6 + 6 * 357 + J * 357])
        i = i + 1
    J = J + 1

print(2)
I = 0
J = 0
K = 0
i = 0
j = 0
k = 0
while J < 569:
    i = 0
    while i < 305:
        j = 0
        while j < 12:
            X_data.append(x1_data2[i + 305 * J + j * 305:(i + 305 * J + j * 305) + 12, :])
            j = j + 1
        i = i + 1
    J = J + 1

I = 0
J = 0
K = 0
i = 0
j = 0
k = 0
while J < 569:
    i = 0
    while i < 305:
        Y_data.append(y1_data2[i + 6 + 6 * 305 + J * 305])
        i = i + 1
    J = J + 1


'''X_data and Y_data consisting of data1 and data2 are organised'''
x_data = np.array(X_data)
'''-1 is the default, 15 is the number of features'''
x_data = x_data.reshape(-1, 15)
x_data = x_data.reshape(-1, 15, 12, 12)
y_data = np.array(Y_data)
y_data = y_data.reshape(-1, 1)
print(x_data)
print(y_data)
'''npy storage'''
outx_file=r'E:\CNN-res\VGG\12_12_datax.npy'
outy_file=r'E:\CNN-res\VGG\12_12_datay.npy'
np.save(outx_file,x_data)
np.save(outy_file,y_data)
print("finish")



print(x_data)
print(y_data)
'''Dataset splitting'''
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size =0.2, random_state=30)
X_test=X_test.reshape(-1,15,12,12)
X_train=X_train.reshape(-1,15,12,12)

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
y_data = np_utils.to_categorical(y_data, num_classes=2)
y_data=np.array(y_data,dtype=np.int8)


counts = np.bincount(y_data[:, 0])
# Calculate category weights based on volume
weight_for_0 = 8. / counts[1]
weight_for_1 = 9. / counts[0]
class_weight = {0: weight_for_0, 1: weight_for_1}
print (class_weight)

'''step6 convolutional neural network model'''
model = Sequential()
model.add(Conv2D(
        32,
        3, # depth
		padding = 'same',
		input_shape=(15,12,12)
						)
		)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(
					pool_size = (2,2),
					strides = (2,2),
					padding = 'same',
					)
		)

##2:128
model.add(Conv2D(128, 3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

##3:256
model.add(Conv2D(256,3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

##4:512
model.add(Conv2D(512, 3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2,padding = 'same'))

##5:512
model.add(Conv2D(512, 3, padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

#####FC
model.add(Flatten())
#model.add(BatchNormalization())
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
########################
adam = Adam(lr = 1e-4)

model.compile(loss = 'binary_crossentropy',optimizer=adam,metrics=['accuracy'])  
model.summary()  

'''step7 model training'''
history=model.fit(
    x_data, y_data,
    epochs=25, batch_size=64,
    validation_data=(x_data, y_data),
    class_weight=class_weight
    )
model.save(r'E:\CNN-res\50_32_17_56.h5')   # HDF5文件，pip install h5py

print('\nSuccessfully saved as a model')


train_history = load_model(r'E:\CNN-res\50_32_17_56.h5',custom_objects={'focal_loss_calc': focal_loss})


# plot training accuracy and loss from history
plt.figure(figsize=(12,9))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy',fontsize=16)
plt.ylabel('accuracy',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train', 'test'],fontsize=16,loc='upper left')
plt.savefig(r'E:\CNN-res\acc50_32_17_56.png')

plt.figure(figsize=(12,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss',fontsize=16)
plt.ylabel('loss',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train', 'test'],fontsize=16, loc='upper right')
plt.savefig(r'E:\CNN-res\loss50_32_17_56.png')
plt.show()

# ROC curves
from sklearn import metrics
import pylab as plt

Font={'size':18, 'family':'Times New Roman'}
y_scores1 = model.predict(X_test)[:, 1]
truelabel = y_test.argmax(axis=-1)

fpr1,tpr1,thres1 = metrics.roc_curve(truelabel, y_scores1,drop_intermediate=False)
roc_auc1 = metrics.auc(fpr1, tpr1)
print(roc_auc1)
plt.figure(figsize=(6,6))
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.4f' % roc_auc1, color='Blue')
plt.legend(loc = 'lower right', prop=Font)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', Font)
plt.xlabel('False Positive Rate', Font)
plt.tick_params(labelsize=15)
plt.savefig(r'E:\CNN-res\roc50_32_17_56.png')
plt.show()

from sklearn.metrics import classification_report
predictions = model.predict_classes(X_test)
print(classification_report(truelabel,predictions))