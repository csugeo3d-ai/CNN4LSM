import numpy as np
import cv2
import os
from skimage import io

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  
#  Get colour dictionary
#  labelFolder Label folders, the reason for traversing the folders is that a label may not contain all the colours of the category
#  classNum Total number of categories (including background)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  Get the name of the file in the folder
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        #ImagePath = labelFolder + "/" + ImageNameList[i]
        ImagePath = labelFolder + ImageNameList[i]
        #img = cv2.imread(r'E:\\huan\\tiffdata\\splitdata\\test\\label\\283.tif')
        img = io.imread(ImagePath).astype(np.uint32)
        #img = cv2.imread(ImagePath).astype(np.uint32)
        #  If greyscale, convert to RGB
        # if(len(img.shape) == 2):
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  To extract unique values, RGB is converted to a number
        img_new = img[:,:] * 1000000 + img[:,:] * 1000 + img[:,:]
        unique = np.unique(img_new)
        #  Add the unique value of the ith pixel matrix to colourDict
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  Take the unique value again for the unique value in the current i pixel matrix
        colorDict = sorted(set(colorDict))
        #  If the number of unique values equals the total number of classes (including background) ClassNum, stop traversing the remaining images
        if(len(colorDict) == classNum):
            break
    #  BGR dictionary storing colours for rendering results at prediction time
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  For the result that does not reach nine digits, the left side of the complementary zero (eg: 5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  First 3 B's, middle 3 G's, last 3 R's.
        color_BGR = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_BGR.append(color_BGR)
    #  Convert to numpy format
    colorDict_BGR = np.array(colorDict_BGR)
    #  GRAY dictionary storing colours for onehot encoding in preprocessing
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1 ,colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY

# def get_weight(numClass, pixel_count):
#    pixel_count = np.zeros((numClass,1))
#    W = 1 / np.log(pixel_count)
#    W = numClass * W / np.sum(W)
#    return W


#################################################################
#  Tagged image folders
LabelPath = r"E:/huan/tiffdatawulingyuan/splitdata/testlabel/"
#  Number of categories (including context)
classNum = 2
#################################################################

#  Get Category Colour Dictionary
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

#  Get all images in a folder
labelList = os.listdir(LabelPath)

#  Reads the first image, whose shape will be used later.
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

#  Number of images
label_num = len(labelList)

#  Put all images in an array
label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label

#  Maps colours to 0,1,2,3...
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i

#  straighten sth. into a dimension
label_all = label_all.flatten()


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(label_all),label_all)
#class_weight = get_weight(numClass=2, pixel_count=label_all)
print(class_weights)