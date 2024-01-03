import numpy as np 
import os
import random
import gdal
import cv2
from skimage import io

#  Get colour dictionary
#  labelFolder label folder, the reason for traversing the folder is that a label may not contain all the colours of the category.
#  classNum Total number of categories (with background)
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
        # If the number of unique values equals the total number of classes (including background) ClassNum, stop traversing the remaining images
        if(len(colorDict) == classNum):
            break
    #  RGB dictionary storing colours for rendering results at prediction time
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  Left zero complement for results that do not reach nine digits(eg:5,201,111->005,201,111)
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
#  fileName Image file name
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

#  Data preprocessing: image normalisation + label onehot coding
#  img image data
#  label Label data
#  classNum Total number of categories (with background)
#  colorDict_GRAY Colour Dictionary
def dataPreprocess(img, label, classNum, colorDict_GRAY):
    #  normalisation
    img = img / 255.0
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    #  Extending data thickness to the classNum layer
    new_label = np.zeros(label.shape + (classNum,))
    #  Turn each category of a flat label into a separate layer
    for i in range(classNum):
        new_label[label == i,i] = 1                                          
    label = new_label
    return (img, label)

#  Training Data Generator
#  batch_size
#  train_image_path Training image paths
#  train_label_path Training Label Paths
#  classNum Total number of categories (including background)
#  colorDict_GRAY colour dictionary
#  resize_shape
def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY, resize_shape = None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + imageList[0])
    #  GDAL read data is (BandNum,Width,Height) to be converted to ->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    #  Unlimited data generation
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if(resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]), np.uint8)
        #  Randomly generate a starting point for a batch
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + imageList[rand + j])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            #  Changing the image size to a specific size
            #  Because resize is not used much, I used the OpenCV implementation, this does not support multiple bands, you can use np for resize if needed.
            if(resize_shape != None):
                img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
            
            img_generator[j] = img
            
            label = readTif(train_label_path + labelList[rand + j]).astype(np.uint8)
            #  If colour, convert to greyscale
            if(len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if(resize_shape != None):
                label = cv2.resize(label, (resize_shape[0], resize_shape[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)
        yield (img_generator,label_generator) 

#  Test Data Generator
#  test_iamge_path Test Data Path
#  resize_shape
def testGenerator(test_iamge_path, resize_shape = None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + imageList[i])
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        #  normalisation
        img = img / 255.0
        if(resize_shape != None):
            #  Changing the image size to a specific size
            img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
        #  Expand the test image by one dimension, consistent with the input [batch_size,img.shape] during training
        img = np.reshape(img, (1, ) + img.shape)
        yield img

#  Save results
#  test_iamge_path Test Data Image Path
#  test_predict_path Test data image prediction result path
#  model_predict Predictive results of the model
#  color_dict colour lexicon
def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis = -1)
        img_out = np.uint8(color_dict[channel_max.astype(np.uint8)])
        #  Modify the difference mode to nearest neighbour difference
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation = cv2.INTER_NEAREST)
        #  Save as lossless compressed png
        cv2.imwrite(test_predict_path + imageList[i][:-4] + ".png", img_out)