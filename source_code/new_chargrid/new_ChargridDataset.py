import numpy as np
import torchvision
from torch.utils.data import Dataset, random_split
import os
from PIL import Image
import torch
from datetime import datetime
import pandas as pd

width = 128
height = 256
nb_classes = 5
nb_anchors = 4  # one per foreground class
base_channels = 64

##########################char###################################
batch_size = 567
input_channels = 61

dir_np_chargrid_cord = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/np_chargrid_cord'
dir_np_chargrid_reduced_resized_1h = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/np_chargrid_reduced_resized_1h'
dir_pd_chargrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/pd_chargrid_reduced_resized'
#

dir_np_chargrid_1h_train='/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/np_chargrids_1h'+'/train'
train_index_filenames = [f for f in os.listdir(dir_np_chargrid_1h_train) if os.path.isfile(os.path.join(dir_np_chargrid_1h_train, f))]
train_index_filenames.sort(key=lambda x: int(x[0:3]))
list_filenames = train_index_filenames
# list_filenames = all_filenames
# for i in range(len(train_index_filenames)):
#     index = int(train_index_filenames[i][0:3])
#     list_filenames.append(all_filenames[index])

# import pdb;pdb.set_trace()
def extract_combined_data(dataset, batch_size):
    print(len(dataset))
    if batch_size > len(dataset):
        raise ValueError('batch_size > length of dataset {}'.format(len(dataset)))

    chargrid_input, bbox_label = [], []
    for i in range(0, batch_size):
        #############################char###############################################
        data = os.path.join(dir_np_chargrid_reduced_resized_1h,dataset[i])
        print(os.path.join(dir_np_chargrid_reduced_resized_1h, dataset[i]))
        chargrid_input.append(data)

        data = os.path.join(dir_np_chargrid_cord, dataset[i])
        bbox_label.append(data)

    return chargrid_input, bbox_label

time_then = datetime.now()

chargrid_input, bbox_label = extract_combined_data(list_filenames, batch_size)
print("total time taken for file parsing: ")
print((datetime.now() - time_then).total_seconds())

class ChargridDataset(Dataset):
    # def __init__(self, chargrid_input, segmentation_ground_truth, anchor_mask_ground_truth, anchor_coordinates):
    def __init__(self, chargrid_input, bbox_label):
        self.chargrid_input = chargrid_input
        self.segmentation_ground_truth = bbox_label
        self.count_now = 0
        # self.anchor_mask_ground_truth = anchor_mask_ground_truth
        # self.anchor_coordinates = anchor_coordinates
        return

    def __len__(self):
        return len(self.chargrid_input)

    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            index = idx.item()
        bbox_gt = np.load(self.segmentation_ground_truth[idx])
        # bbox_gt = np.load(self.segmentation_ground_truth[self.count_now])
        input_chargrid = np.load(self.chargrid_input[idx])
        # input_chargrid = np.load(self.chargrid_input[self.count_now])
        transforms1 = torchvision.transforms.ToTensor()
        # print(self.count_now,self.chargrid_input[idx])
        self.count_now += 1
        # return transforms1(input_chargrid), transforms1(bbox_gt_cord), transforms1(bbox_gt_label)
        return transforms1(input_chargrid), transforms1(bbox_gt)

def get_dataset():
    dataset = ChargridDataset(chargrid_input, bbox_label)
    trainset, testset = random_split(dataset, [len(dataset) - 1, 1])
    print(len(trainset), len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=8)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    return trainloader, testloader

