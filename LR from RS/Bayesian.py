import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from AttResUnet import Attention_ResUNet
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import os
import tensorflow as tf
from skimage import io
import numpy as np
from sklearn.utils import class_weight
from seg_unet import focal
from sklearn.model_selection import GridSearchCV
# from keras.utils.np_utils import *
from tensorflow.keras import utils
import numpy as np 
import os
import random
import gdal
import cv2
from skimage import io
import keras_tuner as kt
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.models import Model
# # from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.layers import merge

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.layers import concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
#from hyperopt import hp
from keras_tuner import HyperParameters, Hyperband, RandomSearch


save_train_image = "E:/huan/tiffdatawulingyuan/splitdata2/traindataset/"
save_train_label = "E:/huan/tiffdatawulingyuan/splitdata2/trainlabel/"
save_validation_image = "E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
save_validation_label = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"

'''
Model-related parameters
'''
#  batch size
batch_size = 4
#  Number of classes (including background)
classNum = 2
#  Model input image size
input_size = (128, 128, 3)
#  Total number of iteration rounds to train the model
epochs = 1
#  Initial learning rate
learning_rate = 0.0001
#  Pre-trained model address
premodel_path = None



#  Get colour dictionary
#  labelFolder Label folders, the reason for traversing the folders is that a label may not contain all the colours of the category
#  classNum Total number of categories (including background)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  Get the name of the file in the folder
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + ImageNameList[i]
        img = io.imread(ImagePath).astype(np.uint32)
        #  If greyscale, convert to RGB
        if(len(img.shape) == 2):
            img = img = 255 * np.array(img).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  To extract unique values, RGB is converted to a number
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  Add the unique value of the ith pixel matrix to colourDict
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  Take the unique value again for the unique value in the current i pixel matrix
        colorDict = sorted(set(colorDict))
        #  If the number of unique values equals the total number of classes (including background) ClassNum, stop traversing the remaining images
        if(len(colorDict) == classNum):
            break
    #  RGB dictionary storing colours for rendering results at prediction time
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  For the result that does not reach nine digits, the left side of the complementary zero (eg: 5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  First 3 R's, middle 3 G's, last 3 B's.
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)
    #  Convert to numpy format
    colorDict_RGB = np.array(colorDict_RGB)
    #  GRAY dictionary storing colours for onehot encoding in preprocessing
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY

#  Read image pixel matrix
#  fileName Image File Name
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

#  Data preprocessing: image normalisation + label onehot coding
#  img image data
#  label Label data
#  classNum Total number of categories (including background)
#  colorDict_GRAY colour dictionary
def dataPreprocess(img, label, classNum, colorDict_GRAY):
    #  normalisation
    imageList = os.listdir(img)
    labelList = os.listdir(label)
    img = readTif(img + imageList[0])
    #  GDAL read data is (BandNum,Width,Height) to be converted to ->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    label = readTif(label + labelList[0]).astype(np.uint8)
    label = label.swapaxes(1, 0)
    label = label.swapaxes(1, 2)
    img = img / 255.0
    label =label / 255.0
    # for i in range(colorDict_GRAY.shape[0]):
    #     label[label == colorDict_GRAY[i][0]] = i
    #  Extending data thickness to the classNum layer
    new_label = np.zeros(label.shape + (classNum,))
    #  Turn each category of a flat label into a separate layer
    for i in range(classNum):
        new_label[label == i,i] = 1                                          
    label = new_label
    return (img, label)

#  Number of training data
train_num = len(os.listdir(save_train_image))
#  Number of validation data
validation_num = len(os.listdir(save_validation_image))
#  How many batch_sizes per epoch in the training set
steps_per_epoch = train_num / batch_size
#  How many batch_sizes per epoch for the validation set
validation_steps = validation_num / batch_size
#  Colour dictionary for labels, used for onehot coding
colorDict_RGB, colorDict_GRAY = color_dict(save_train_label, classNum)

# x_train,y_train = dataPreprocess(save_train_image, save_train_label, classNum, colorDict_GRAY)

# x_test,y_test = dataPreprocess(save_validation_image, save_validation_label, classNum, colorDict_GRAY)

#  Get a generator that generates training data at the rate of batch_size
train_Generator = trainGenerator(batch_size,
                                 save_train_image, 
                                 save_train_label,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  Get a generator that generates validation data at the rate of batch_size
validation_data = trainGenerator(batch_size,
                                 save_validation_image,
                                 save_validation_label,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

# tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(1)
# te_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(1)
# x_train = cv2.resize(np.array(x_train), (128, 128))
# y_train = cv2.resize(np.array(y_train), (128, 128))
# x_test = cv2.resize(np.array(x_test), (128, 128))
# y_test = cv2.resize(np.array(y_test), (128, 128))
# x_train = np.expand_dims(x_train,axis=0)
# y_train = np.expand_dims(y_train,axis=0)
# x_test = np.expand_dims(x_test,axis=0)
# y_test = np.expand_dims(y_test,axis=0)
# x_train = save_train_image
# y_train = save_train_label
# x_test = save_validation_image
# y_test = save_validation_label

# # normalize pixels to range 0-1
# train_x = x_train / 255.0
# test_x = x_test / 255.0

# #one-hot encode target variable
# train_y = to_categorical(y_train)
# test_y = to_categorical(y_test)

# print(x_train.shape) #(57000, 28, 28)
# print(y_train.shape) #(57000, 10)
# print(x_test.shape) #(10000, 28, 28)
# print(y_test.shape) #(10000, 10)

print(1)

def build_model(hp):
    model = Sequential()
    inputs = Input(input_size)
    #  2D Convolutional Layer
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1))
    #  For maximum pooling of spatial data
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4))
    #  Dropout regularisation to prevent overfitting
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
 
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  Upsampling followed by convolution is equivalent to the transpose-convolution operation
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    try:
        merge6 = concatenate([drop4,up6],axis = 3)
    except:
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6))
 
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    try:
        merge7 = concatenate([conv3,up7],axis = 3)
    except:
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7))
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    try:
        merge8 = concatenate([conv2,up8],axis = 3)
    except:
        merge8 = merge([conv2,up8],mode = 'concat', concat_axis = 3)
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8))
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    try:
        merge9 = concatenate([conv1,up9],axis = 3)
    except:
        merge9 = merge([conv1,up9],mode = 'concat', concat_axis = 3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation = 'sigmoid')(conv9)
 
    model = Model(inputs = inputs, outputs = conv10)
 
    #  For configuring training models (optimiser, objective function, model evaluation criteria)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    hp_optimizer=hp.Choice('Optimizer', values=['Adam', 'SGD'])

    if hp_optimizer == 'Adam':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    elif hp_optimizer == 'SGD':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        nesterov=True
        momentum=0.9
    weights = {0: 0.50454072, 1: 55.55734012}
    #hp = HyperParameters()
    model.compile(optimizer=hp_optimizer,
                  loss = focal(alpha = hp.Float('alpha', min_value=0,max_value=1), gamma = hp.Float('gamma', min_value=0,max_value=5)),
                  metrics=['accuracy'])
    # model.compile(optimizer= Adam(lr = learning_rate), loss = focal(), metrics=['accuracy'])
    return model
hp = HyperParameters()
#Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4]))
#hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
#loss = focal(alpha = hp.Float('alpha', min_value=0,max_value=1), gamma = hp.Float('gamma', min_value=0,max_value=5))
# model.compile(optimizer = hp_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
#tuner_mlp = kt.tuners.BayesianOptimization(model, objective='val_loss', max_trials=30, directory='.', project_name='tuning-mlp')
#tuner_mlp.search(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=early_stopping)

tuner = RandomSearch(
    build_model,
    # `tune_new_entries=False` prevents unlisted parameters from being tuned
    objective='val_accuracy',
    max_trials=30,
    directory='my_dir50',
    project_name='50')

# tuner.search(x_train, y_train[:,:-1,],
#              validation_data=(x_test, y_test))
tuner.search(train_Generator,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            callbacks = [early_stopping],
            validation_data = validation_data,
            validation_steps = validation_steps)
tuner.search_space_summary()
models = tuner.get_best_models(num_models=2)
tuner.results_summary()
"""
Keras Tuner Four regulators are provided：
RandomSearch、Hyperband、BayesianOptimization和Sklearn
"""
# #Instantiate the regulator and perform an override
# tuner = kt.Hyperband(model,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='intro_to_kt')


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


#Training Models
"""
Use hyperparameters obtained from search to find the optimal number of lifecycles to train the model。
"""
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#The optimal number of cycles is no longer trained
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hyper = model.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

#Evaluating supermodels on test data
eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)