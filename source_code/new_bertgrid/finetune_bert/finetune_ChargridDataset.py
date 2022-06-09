import numpy as np
import torchvision
from skimage.transform import resize
from torch.utils.data import Dataset, random_split
import os
from PIL import Image
import torch
from datetime import datetime
import pandas as pd



nb_classes = 5
nb_anchors = 4  # one per foreground class
# input_channels = 61
input_channels = 768
base_channels = 64
file_lenth = 555

dir_np_chargrid_cord = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/np_bertgrid_cord'
dir_np_chargrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/np_bertgrid_reduced_resized'
dir_pd_chargrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/pd_bertgrid_reduced_resized'
dir_input_ids = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/finetune_preprocessing/input_ids'

dir_np_chargrid_1h_train='/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/np_chargrids_1h'+'/train'+'/done'
train_index_filenames = [f for f in os.listdir(dir_np_chargrid_1h_train) if os.path.isfile(os.path.join(dir_np_chargrid_1h_train, f))]
train_index_filenames.sort(key=lambda x: int(x[0:3]))
list_filenames = train_index_filenames

def extract_combined_data(list_filenames, file_lenth):
    print(len(list_filenames))
    if file_lenth > len(list_filenames):
        raise ValueError('file_lenth > length of dataset {}'.format(len(list_filenames)))
    chargrid_input = []
    bbox_label = []
    input_ids = []
    for i in range(0, file_lenth):

        data = os.path.join(dir_np_chargrid_cord, list_filenames[i])
        bbox_label.append(data)

        data = os.path.join(dir_input_ids,list_filenames[i])
        input_ids.append(data)

    return  bbox_label, input_ids

time_then = datetime.now()

bbox_label, input_ids= extract_combined_data(list_filenames, file_lenth)
print("total time taken for file parsing: ")
print((datetime.now() - time_then).total_seconds())

class ChargridDataset(Dataset):
    def __init__(self,bbox_label, input_ids):
        self.bbox_label = bbox_label
        self.input_ids = input_ids
        self.count_now = 0
        self.count_next = 0
        return

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            index = idx.item()
        self.count_now = self.count_next
        # bbox_label = self.bbox_label[idx]
        # input_ids = self.input_ids[idx]

        bbox_label = np.load(self.bbox_label[self.count_now])
        input_ids = np.load(self.input_ids[self.count_now])
        transforms = torchvision.transforms.ToTensor()
        self.count_next += 1
        # return transforms(bbox_label), transforms(input_ids)
        return bbox_label, input_ids

def get_dataset():
    dataset = ChargridDataset(bbox_label, input_ids)
    test_no = int(len(dataset) * 0.01)
    trainset, testset = random_split(dataset, [len(dataset) - test_no, test_no])

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=8)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4)

    return trainloader, testloader

