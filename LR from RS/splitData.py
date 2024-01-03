# *_*coding: utf-8 *_*
# Author --LiMing--

import os
import random
import shutil
import time

def copyFile(dataFileDir, labelFileDir):
    data_image_list = os.listdir(dataFileDir) # Get the original path of the image
    label_image_list = os.listdir(labelFileDir) # Get the original path of the image
    data_image_list.sort(key=lambda x: int(x.split('.')[0]))
    label_image_list.sort(key=lambda x: int(x.split('.')[0]))

    image_number = len(data_image_list)
    train_number = int(image_number * train_rate)
    train_sample = random.sample(data_image_list, train_number) # Get a random 0.8 scale image from image_list.
    test_sample = list(set(data_image_list) - set(train_sample))
    sample = [train_sample, test_sample]

    # Copying images to the destination folder
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k] + 'dataset'):
            for name in sample[k]:
                shutil.copy(os.path.join(dataFileDir, name), os.path.join(save_dir[k] + 'dataset'+'/', name))
        else:
            os.makedirs(save_dir[k] + 'dataset')
            for name in sample[k]:
                shutil.copy(os.path.join(dataFileDir, name), os.path.join(save_dir[k] + 'dataset'+'/', name))
        if os.path.isdir(save_dir[k] + 'label'):
            for name in sample[k]:
                shutil.copy(os.path.join(labelFileDir, name), os.path.join(save_dir[k] + 'label'+'/', name))
        else:
            os.makedirs(save_dir[k] + 'label')
            for name in sample[k]:
                shutil.copy(os.path.join(labelFileDir, name), os.path.join(save_dir[k] + 'label'+'/', name))

if __name__ == '__main__':
    time_start = time.time()

    # Original dataset path
    origion_path = 'E:/huan/tiffdatawulingyuan/smalldata/fourth/'

    # Save Path
    save_train_dir = 'E:/huan/tiffdatawulingyuan/splitdata4/train'
    save_test_dir = 'E:/huan/tiffdatawulingyuan/splitdata4/test'
    save_dir = [save_train_dir, save_test_dir]

    # Training set ratio
    train_rate = 0.8

    # Type and number of data sets
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    dataFileDir=os.path.join(origion_path, 'dataset')
    labelFileDir=os.path.join(origion_path, 'label')
    copyFile(dataFileDir, labelFileDir)
    print('end of division！')
    # for i in range(num_classes):
    #     class_name = file_list[i]
    #     image_Dir = os.path.join(origion_path, class_name)
    #     copyFile(image_Dir, class_name)
        

    time_end = time.time()
    print('---------------')
    print('Total time spent on training and test set partitioning%s!' % (time_end - time_start))