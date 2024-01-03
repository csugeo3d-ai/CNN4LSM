import os
from tqdm import tqdm
import numpy as np

 
def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']##Here sample['label'] returns the label image's lable mask
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)##Counting the number of different classes of pixels in each image
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))##Here are the weights calculated for each category pixel
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(os.path.dirname(dataset), dataset+'_classes_weights.npy')##Generate weights file
    np.save(classes_weights_path, ret)##Save the pixel weights for each category to a file
 
    return ret

#  Training data labelling path
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata/trainlabel/"
dataset1 = "E:/huan/tiffdatawulingyuan/splitdata/model/"
weight = calculate_weigths_labels(dataset1, train_label_path, 2)

#criterion=nn.CrossEntropyLoss(weight=self.weight,ignore_index=self.ignore_index, reduction='mean')