import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet
from AttResUnet import Attention_ResUNet
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import os
# from skimage import io
import numpy as np
from sklearn.utils import class_weight


'''
Dataset related parameters
'''
#  Training data image path
train_image_path = r"E:/tiffdatawulingyuan/splitdata2/traindataset/"
#  Training data labelling path
train_label_path = r"E:/tiffdatawulingyuan/splitdata2/trainlabel/"
#  Verify the data image path
validation_image_path = r"E:/tiffdatawulingyuan/splitdata2/testdataset/"
#  Validating data tag paths
validation_label_path = r"E:/tiffdatawulingyuan/splitdata2/testlabel/"

'''
Model-related parameters
'''
#  batch_size
batch_size = 4
#  Number of classes (including background)
classNum = 2
#  Model input image size
input_size = (128, 128, 3)
#  Total number of iteration rounds to train the model
# epochs = 20
epochs = 50
#  Initial learning rate
# learning_rate = 0.0001
learning_rate = 0.01
#  Pre-trained model address
premodel_path = None
#  Training model save address
model_path = r"E:\U-Net\model\attresunet_model_epoch20_1e-4_focaldataset4.h5"

#  Number of training data
train_num = len(os.listdir(train_image_path))
#  Number of validation data
validation_num = len(os.listdir(validation_image_path))
#  How many batch_sizes per epoch in the training set
steps_per_epoch = train_num / batch_size
#  How many batch_sizes per epoch for the validation set
validation_steps = validation_num / batch_size
#  Colour dictionary for labels, used for onehot coding
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  Get a generator that generates training data at the rate of batch_size
train_Generator = trainGenerator(batch_size,
                                 train_image_path, 
                                 train_label_path,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  Get a generator that generates validation data at the rate of batch_size
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  Defining the model
model = Attention_ResUNet( input_size = input_size, 
             classNum = classNum, 
             learning_rate = learning_rate)

model.summary()
#  callback function
#  val_loss stops training if there is no drop for 10 consecutive rounds
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
#  When 3 epochs pass without val_loss decreasing, the learning rate is halved
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1)
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor = 'loss',
                                   verbose = 1,# Log display mode:0->quiet mode,1->progress bar,2->one line per turn
                                   save_best_only = True)

#  Get current time
start_time = datetime.datetime.now()

weights = {0: 0.50454072, 1: 55.55734012}

#  model training
history = model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks = [early_stopping, model_checkpoint, model_checkpoint],
                    validation_data = validation_data,
                    validation_steps = validation_steps)

#  Total training time
end_time = datetime.datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
print(log_time)
with open('TrainTime_%s.txt'%time,'w') as f:
    f.write(log_time)
    
#  Save and plot loss,acc
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(acc)):
    sheet.write(i, 0, acc[i])
    sheet.write(i, 1, val_acc[i])
    sheet.write(i, 2, loss[i])
    sheet.write(i, 3, val_loss[i])
book.save(r'AccAndLoss_%s.xls'%time)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy_%s.png"%time, dpi = 300)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss_%s.png"%time, dpi = 300)
plt.show()

