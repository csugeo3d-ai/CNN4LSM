import os
import gdal
import numpy as np

#  Read tif dataset
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "File cannot be opened")
    return dataset
    
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
        dataset.SetGeoTransform(im_geotrans) #Write the affine transformation parameters
        dataset.SetProjection(im_proj) #Write to projection
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

# Pixel coordinates and geographic coordinates affine transformations
def CoordTransf(Xpixel, Ypixel, GeoTransform):
    XGeo = GeoTransform[0]+GeoTransform[1]*Xpixel+Ypixel*GeoTransform[2];
    YGeo = GeoTransform[3]+GeoTransform[4]*Xpixel+Ypixel*GeoTransform[5];
    return XGeo, YGeo    

'''
Sliding Window Cropping Functions
TifPath image path
SavePath Save directory after cropping
CropSize Cutting size
RepetitionRate repetition rate
'''
def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)#Getting data
    
    #  Get the number of files in the current folder len, and name the image to be cropped with len+1.
    new_name = len(os.listdir(SavePath)) + 1
    #  Crop the image, the repetition rate is RepetitionRate
    
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  If the image is a single band
            if(len(img.shape) == 2):
                cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  If the image is multi-band
            else:
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  Write images
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
            #  File name + 1
            new_name = new_name + 1
    #  Crop the last column forward
    for i in range(int((height-CropSize*RepetitionRate)/(CropSize*(1-RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        else:
            cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        #  Write images
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        new_name = new_name + 1
    #  Crop the last row forward
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[(height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                          (height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        #  File name + 1
        new_name = new_name + 1
    #  Crop lower right corner
    if(len(img.shape) == 2):
        cropped = img[(height - CropSize) : height,
                      (width - CropSize) : width]
    else:
        cropped = img[:,
                      (height - CropSize) : height,
                      (width - CropSize) : width]
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
    new_name = new_name + 1
     
#  Crop image 1 to a 256×256 dataset with a repetition rate of 0.5
TifCrop(r"E:/huan/tiffdatawulingyuan/originaldata/fourth/data.tif",
        r"E:/huan/tiffdatawulingyuan/smalldata/fourth/dataset", 128, 0.5)
TifCrop(r"E:/huan/tiffdatawulingyuan/originaldata/fourth/label.tif",
        r"E:/huan/tiffdatawulingyuan/smalldata/fourth/label", 128, 0.5)