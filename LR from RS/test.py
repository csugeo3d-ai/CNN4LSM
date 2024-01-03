import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet

from dataProcess import testGenerator, saveResult, color_dict


#  Training model save address
model_path = r"E:/huan/tiffdatawulingyuan/model/unet_model_epoch20_1e-4_focaldataset3.h5"
#  Test Data Path
test_iamge_path = r"E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
#  Result Save Path
save_path = r"E:/huan/tiffdatawulingyuan/predictdata/"
#  Number of test data
test_num = len(os.listdir(test_iamge_path))
#  Number of classes (including background)
classNum = 2
#  Model input image size
input_size = (128, 128, 3)
#  Generate image size
output_size = (128, 128)
#  Training data labelling path
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"
#  Colour dictionary for labels
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  Numpy array of predicted values
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  Save results
saveResult(test_iamge_path, save_path, results, colorDict_GRAY, output_size)