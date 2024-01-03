import gdal
import numpy as np
from tensorflow.keras.models import load_model
# from keras import losses
import datetime
import math
import sys
from seg_unet import unet

#  Read tif dataset
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "File cannot be opened")
    #  Number of columns in the raster matrix
    width = dataset.RasterXSize 
    #  Number of rows of the raster matrix
    height = dataset.RasterYSize 
    #  Number of bands
    bands = dataset.RasterCount 
    #  Getting data
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  Get affine matrix information
    geotrans = dataset.GetGeoTransform()
    #  Getting Projection Information
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

#  Save tif file function
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #Creating Documents
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

#  tif crop (tif pixel data, crop side length)
def TifCroppingArray(img, SideLength):
    #  chained list of cuts
    TifArrayReturn = []
    #  Number of image blocks on column
    ColumnNum = int((img.shape[0] - SideLength * 2) / (128 - SideLength * 2))
    #  Number of image blocks on line
    RowNum = int((img.shape[1] - SideLength * 2) / (128 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (128 - SideLength * 2) : i * (128 - SideLength * 2) + 128,
                          j * (128 - SideLength * 2) : j * (128 - SideLength * 2) + 128]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  Crop one row and one column forward, taking into account that there will be leftovers in the rows and columns
    #  Crop the last column forward
    for i in range(ColumnNum):
        cropped = img[i * (128 - SideLength * 2) : i * (128 - SideLength * 2) + 128,
                      (img.shape[1] - 128) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  Crop the last row forward
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 128) : img.shape[0],
                      j * (128-SideLength*2) : j * (128 - SideLength * 2) + 128]
        TifArray.append(cropped)
    #  Crop forward to lower right corner
    cropped = img[(img.shape[0] - 128) : img.shape[0],
                  (img.shape[1] - 128) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  Remaining number of columns
    ColumnOver = (img.shape[0] - SideLength * 2) % (128 - SideLength * 2) + SideLength
    #  Remaining number of rows
    RowOver = (img.shape[1] - SideLength * 2) % (128 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

#  Label visualisation, i.e. assigning an n-value to the nth class
def labelVisualize(img):
    img_out = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #  Assign a value of n to the nth class
            img_out[i][j] = np.argmax(img[i][j])
    return img_out

#  Normalise the test image and make it dimensionally consistent with the training image
def testGenerator(TifArray):
    for i in range(len(TifArray)):
        for j in range(len(TifArray[0])):
            img = TifArray[i][j]
            #  normalisation
            img = img / 255.0
            #  Change the shape without changing the content of the data.
            img = np.reshape(img,(1,)+img.shape)
            yield img

#  Obtaining the results matrix
def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    # j to mark the number of lines
    j = 0  
    for i,item in enumerate(npyfile):
        img = labelVisualize(item)
        img = img.astype(np.uint8)
        #  Special considerations for the leftmost column, with the left edge spliced in
        if(i % len(TifArray[0]) == 0):
            #  The first row's to be given special consideration again, with the top edge taken into account
            if(j == 0):
                result[0 : 128 - RepetitiveLength, 0 : 128-RepetitiveLength] = img[0 : 128 - RepetitiveLength, 0 : 128 - RepetitiveLength]
            #  The last line of the last line has to be given special consideration again, with the lower edge taken into account
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 128 - RepetitiveLength] = img[128 - ColumnOver - RepetitiveLength : 128, 0 : 128 - RepetitiveLength]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:128-RepetitiveLength] = img[RepetitiveLength : 128 - RepetitiveLength, 0 : 128 - RepetitiveLength]   
        #  Special considerations for the rightmost column, with the right edge to be spliced in
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  The first row's to be given special consideration again, with the top edge taken into account
            if(j == 0):
                result[0 : 128 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 128 - RepetitiveLength, 128 -  RowOver: 128]
            #  The last line of the last line has to be given special consideration again, with the lower edge taken into account
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[128 - ColumnOver : 128, 128 - RowOver : 128]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 128 - RepetitiveLength, 128 - RowOver : 128]   
            #  Finish the rightmost side of each row, +1 for the number of rows
            j = j + 1
        #  Not the leftmost nor the rightmost case
        else:
            #  Special consideration should be given to the first row, with the top edge taken into account
            if(j == 0):
                result[0 : 128 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 128 - RepetitiveLength, RepetitiveLength : 128 - RepetitiveLength]         
            #  The last row of the last line has to be taken into special consideration, the lower edge has to be taken into account
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[128 - ColumnOver : 128, RepetitiveLength : 128 - RepetitiveLength]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 128 - RepetitiveLength, RepetitiveLength : 128 - RepetitiveLength]
    return result

area_perc = 0.5
TifPath = r"E:/huan/tiffdatawulingyuan/originaldata/zzjbigarea.tif"
ModelPath = r"E:/huan/tiffdatawulingyuan/model/unet_modelwly_epoch20_1e-4_weight.h5"
ResultPath = r"E:/huan/tiffdatawulingyuan/bigareapre_result12.tif"

RepetitiveLength = int((1 - math.sqrt(area_perc)) * 128 / 2)

#  Record test consumption time
testtime = []
#  Get current time
starttime = datetime.datetime.now()

im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(TifPath)
im_data = im_data.swapaxes(1, 0)
im_data = im_data.swapaxes(1, 2)

TifArray, RowOver, ColumnOver = TifCroppingArray(im_data, RepetitiveLength)
endtime = datetime.datetime.now()
text = "Read tif and crop preprocessing complete, current time: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

model = unet(ModelPath)
testGene = testGenerator(TifArray)
results = model.predict_generator(testGene,
                                  len(TifArray) * len(TifArray[0]),
                                  verbose = 1)
endtime = datetime.datetime.now()
text = "Model predictions are complete. Currently taking time.: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

#Save results
result_shape = (im_data.shape[0], im_data.shape[1])
result_data = Result(result_shape, TifArray, results, 2, RepetitiveLength, RowOver, ColumnOver)
writeTiff(result_data, im_geotrans, im_proj, ResultPath)
endtime = datetime.datetime.now()
text = "The result is spliced. It's taking time: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
with open('timelog_%s.txt'%time, 'w') as f:
    for i in range(len(testtime)):
        f.write(testtime[i])
        f.write("\r\n")