import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np

import new_ChargridDataset
from new_ChargridNetwork import new_ChargridNetwork
from datetime import datetime

from torchsummary import summary

# os.environ['CUDA_VISIBLE_DEVICE']='3'
file_str = 'new_bertgrid_max_pool_sep_new_weight'
device0 = torch.device("cuda:2")
# device0 = torch.device("cuda")
num_epochs = 100
EPOCH = 70

class_weight = torch.tensor([[0.8311], [0.0339], [0.0192], [0.1031], [0.0127]])
constant_weight = 1.04
log_weight = 1/torch.log(class_weight + constant_weight).to(device0)
# EPOCH = 574

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)


def init_weights_in_last_layers(net):
    torch.nn.init.constant_(net.ssd_d_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_e_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_f_block[6].weight, 1e-3)

if __name__ == '__main__':
    torch.manual_seed(0)
    HW = 768
    # HW = 127-32+1
    C = 64
    num_classes = 5
    num_anchors = 4

    trainloader, testloader = new_ChargridDataset.get_dataset()

    net = new_ChargridNetwork(HW, C, num_classes, num_anchors)

    model_dir = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/model_output/new_bertgrid_max_pool_sep_new_weight'

    net = net.apply(init_weights)
    init_weights_in_last_layers(net)
    net = net.to(device0)
    optimizer_ft = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer_ft = optim.Adam(net.parameters(), lr=0.005)
    losses = {
        'loss1': [],
        'loss2': [],
        'loss3': [],
        'combined_losses': []
    }

    checkpoint = torch.load(os.path.join(model_dir, 'epoch-'+file_str+'-{}.pt'.format(EPOCH)), map_location=device0)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    load_epoch = checkpoint['epoch']
    losses = checkpoint['loss']
########################################################################################
    # loss1 = FocalLoss2d()
    # loss1 = nn.CrossEntropyLoss()
    loss2 = nn.BCELoss()
    loss3 = nn.SmoothL1Loss()
    loss4 = nn.CrossEntropyLoss(weight=log_weight)
    sum_label=torch.zeros((5))
    for epoch in range(num_epochs-load_epoch):
    # for epoch in range(num_epochs):

        final_loss = 0.0
        #num_epo = epoch
        #print(num_epo)
        num = 0
        # for inputs, label1, label2, label3 in trainloader:
        for inputs, label1, count_now in trainloader:
            num = num + 1
            inputs = inputs.to(device0)
            label1 = label1.to(device0)
            optimizer_ft.zero_grad()
            time_then = datetime.now()

            output1, output2, output3 = net(inputs.float())
            print(output1[0][0][0][0]," total time taken for forward passing: ")
            print(num, (datetime.now() - time_then).total_seconds())

            loss_1 = 0
            softmax = nn.Softmax(dim=0)
            for i in range(output1.shape[0]):
                output_temp = output1[i,:,:,:]
                label_temp = label1[i,:,:,:].reshape(5,-1)
                gt_cord = label_temp[:4, :]
                gt_label = label_temp[4, :]
                gt_cord = gt_cord[:, :int(torch.nonzero(gt_label>-1,as_tuple=False)[-1]+1)]
                gt_cord = gt_cord.int()
                gt_label = gt_label[:int(torch.nonzero(gt_label>-1,as_tuple=False)[-1]+1)]
                gt_label = gt_label.int()
                # import pdb;pdb.set_trace()
                predicted_label = torch.zeros((1,output1.shape[1],gt_cord.shape[1])).to(device0)
                for j in range(gt_cord.shape[1]):
                    bbox = output_temp[:,gt_cord[2,j]:gt_cord[3,j],gt_cord[0,j]:gt_cord[1,j]]
                    if bbox.shape[1]==0 or bbox.shape[2]==0:
                        import pdb;pdb.set_trace()
                    output_bbox_pool = torch.nn.functional.max_pool2d(bbox, kernel_size = (bbox.shape[1],bbox.shape[2])).reshape(5)

                    predicted_label[:,:,j] = output_bbox_pool
                # import pdb;pdb.set_trace()
                gt_label_one_hot = torch.zeros((1,output1.shape[1],gt_cord.shape[1])).to(device0)
                gt_label_one_hot[0,:,:] = torch.nn.functional.one_hot(gt_label.long(), 5).permute(1,0)
                loss_1 += loss4(predicted_label.permute(2,1,0).view(gt_cord.shape[1], output1.shape[1]), gt_label.long())/output1.shape[0]            # print("EPOCH: {},NUM: {}, loss1: {}".format(epoch, num,loss_1))
            print("EPOCH: {},NUM: {}, loss1: {}".format(epoch+load_epoch+1, num,loss_1))
            losses['loss1'].append(loss_1.item())
            final_loss = loss_1 #+ loss_2 + loss_3
            losses['combined_losses'].append(final_loss.item())
            time_then = datetime.now()
            final_loss.backward()
            optimizer_ft.step()

            print("total time taken for backwards: ")
            print((datetime.now() - time_then).total_seconds())

        print("Epoch {}/{}, Loss: {:.3f}".format(epoch+load_epoch+1, num_epochs, final_loss.item()))
        print(os.path.join(model_dir, 'epoch-'+file_str+'{}.pt'.format(epoch+load_epoch+1)))
        torch.save({
            'epoch': epoch+load_epoch+1,
            # 'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'loss': losses,
        }, os.path.join(model_dir, 'epoch-'+file_str+'-{}.pt'.format(epoch+load_epoch+1)))


    print('================')
    print('================')
    print('loss1: ' + str(losses['loss1']))
    print('combined: ' + str(losses['combined_losses']))
    print('Finished Training')
