from PIL import Image
from osgeo import gdal
import os
import json

# Take tiff to png as an example, other formats are the same.
# Just change the path in the code to your own image storage path.

imagesDirectory= r"E:/huan/tiffdata/splitdata/test/label"  # Path to the folder where the tiff images are located
distDirectory = os.path.dirname(imagesDirectory)
distDirectory = os.path.join(distDirectory, "E:/huan/tiffdata/splitdata/test/labelpng")# Path to the folder where you want to store the png format
for imageName in os.listdir(imagesDirectory):
    imagePath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagePath)# Open tiff image
    distImagePath = os.path.join(distDirectory, imageName[:-4]+'.png')# Change the image extension to .png with the same name as the original image
    if image.mode == "F":
        image = image.convert('RGB')
    image.save(distImagePath)# Save png image

imagesDirectory1= r"E:/huan/tiffdatawulingyuan/predictdata"  # Path to the folder where the tiff images are located
distDirectory1 = os.path.dirname(imagesDirectory1)
distDirectory1 = os.path.join(distDirectory1, "E:/huan/tiffdata/splitdata/test/predict1png")# Path to the folder where you want to store the png format
for imageName in os.listdir(imagesDirectory1):
    imagePath = os.path.join(imagesDirectory1, imageName)
    image = Image.open(imagePath)# Open tiff image
    distImagePath = os.path.join(distDirectory1, imageName[:-4]+'.png')# Change the image extension to .png with the same name as the original image
    if image.mode == "F":
        image = image.convert('RGB')
    image.save(distImagePath)# Save png image

# Get the four values of the confusion matrix (in the case of road extraction, the road region [255,255,255] and the background region [0,0,0])
# TP: Positive samples predicted by the model to be positive classes (predicted roads and labelled roads)
# TN: Negative samples predicted by the model to be in the negative category (predicted background and true background)
# FP: Negative samples predicted by the model to be in the positive category (predicted roads but true context)
# FN: Positive samples predicted by the model to be in the negative category (predicted background but real roads)
def get_vaslue(predict_folders_path, label_folders_path):
    #  Tagged image folders
    label_folders_path = r"E:/huan/tiffdata/splitdata/test/labelpng/"
    #  Forecast image folder
    predict_folders_path = r"E:/huan/tiffdata/splitdata/test/predict1png/"
    #################################################################
    predict_folders = os.listdir(predict_folders_path)
    label_folders = os.listdir(label_folders_path)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for folder in predict_folders:
        # Get Image Path
        predict_folder_path = os.path.join(predict_folders_path, folder)
        label_folder_path = os.path.join(label_folders_path, folder)
        # Load image and assign four channels
        predict = Image.open(predict_folder_path)
        predict = predict.convert('RGBA')
        label = Image.open(label_folder_path)
        label = label.convert('RGBA')
        heigh, width = predict.size
        # save_name = str(folder).split('.')[0]
        for i in range(heigh):
            for j in range(width):
                r_1, g_1, b_1, a_1 = predict.getpixel((i, j))
                r_2, g_2, b_2, a_2 = label.getpixel((i, j))
                if r_1 == 255:
                    if r_2 == 255:
                        TP += 1
                    if r_2 == 0:
                        FP += 1
                if r_1 == 0:
                    if r_2 == 255:
                        FN += 1
                    if r_2 == 0:
                        TN += 1
    return float(TP), float(TN), float(FP), float(FN)


# list to txt
def list2txt(list, save_path, txt_name):
    with open(save_path + r'/' + txt_name, 'w') as f:
        json.dump(list, f)


def evoluation(TP, TN, FP, FN):
    evo = []
    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision
    #precision = TP / (TP + FP)
    # recall
    recall = TP / (TP + FN)
    # miou
    miou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    # F1
    #f1 = 2 * ((precision * recall) / (precision + recall))
    evo.append('accuracy:{}  recall:{}  miou:{}'.format(accuracy, recall, miou))
    print(evo)
    list2txt(evo, r"", '')
    return evo


if __name__ == '__main__':
    predict_path = r''
    label_path = r''
    TP, TN, FP, FN = get_vaslue(predict_path, label_path)
    evoluation(TP, TN, FP, FN)
