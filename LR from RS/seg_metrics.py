import numpy as np
import cv2
import os
from skimage import io

""" 
confusion matrix 
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

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  Return Confusion Matrix
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  Returns the overall pixel precision OA of all classes
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  Returns the precision of all categories precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    #  Return the recall of all categoriesrecall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  Return to Intersection and Combination Ratio IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  Return the average intersection ratio mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  Return frequency intersection ratio FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#################################################################
#  Tagged image folders
LabelPath = r"E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"
#  Forecast image folder
PredictPath = r"E:/huan/tiffdatawulingyuan/predictdata/"
#  Number of categories (including context)
classNum = 2
#################################################################
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

#  Get all images in a folder
labelList = os.listdir(LabelPath)
PredictList = os.listdir(PredictPath)

#  Reads the first image, whose shape will be used later.
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

#  Number of images
label_num = len(labelList)

#  Put all images in an array
label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label
    Predict = cv2.imread(PredictPath + "//" + PredictList[i])
    Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
    predict_all[i] = Predict

#  Maps colours to 0,1,2,3...
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i
    predict_all[predict_all == colorDict_GRAY[i][0]] = i

#  straighten sth. into a dimension
label_all = label_all.flatten()
predict_all = predict_all.flatten()

#  Calculate the confusion matrix and each precision parameter
confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)

for i in range(colorDict_BGR.shape[0]):
    #  Output category colours, need to install webcolors, direct pip install webcolors
    try:
        import webcolors
        rgb = colorDict_BGR[i]
        rgb[0], rgb[2] = rgb[2], rgb[0]
        print(webcolors.rgb_to_name(rgb), end = "  ")
    #  Outputs grey scale values if not installed.
    except:
        print(colorDict_GRAY[i][0], end = "  ")
print("")
print("confusion matrix:")
print(confusionMatrix)
print("precision:")
print(precision)
print("recall:")
print(recall)
print("F1-Score:")
print(f1ccore)
print("OA:")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)