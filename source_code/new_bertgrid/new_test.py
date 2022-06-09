import os
import torch.nn.modules as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
##############################################
from torchvision import transforms
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


###############################################
import new_ChargridDataset_test
from new_ChargridNetwork import new_ChargridNetwork
from datetime import datetime
import pandas as pd
import json

EPOCH = 50
model_dir = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/model_output/new_bertgrid_max_pool_sep_new_weight'
file_str = 'new_bertgrid_max_pool_sep_new_weight'
device0 = torch.device("cuda:1")
LR = 0.001
dir_classes='/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/gt_classes/key'
out_dir_img = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/img/new_bert_lr001_no_arg_max_pool_sep_filter_new_weight/'
# ([[0.8311], [0.0339], [0.0192], [0.1031], [0.0127]])
class_weight = torch.tensor([[0.8311], [0.0339], [0.0192], [0.1031], [0.0127]])
constant_weight = 1.04
log_weight = 1/torch.log(class_weight + constant_weight).to(device0)

def input_transform(crop_size):
    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)


# 加载模型，解决命名和维度不匹配问题,解决多个gpu并行
def load_state_keywise(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device0)
    pretrained_dict = pretrained_dict['model_state_dict']
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith('module.')):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def plot_loss(checkpoint):
    y1 = []
    y2 = []
    y3 = []

    #or i in range(0,n):
    #print('EPOCH: {}'.format(i))
    # checkpoint = torch.load(os.path.join(model_dir, 'epoch-{}.pt'.format(EPOCH)))
    losses = checkpoint['loss']

    loss1 = list(losses['loss1'])
    loss2 = list(losses['loss2'])
    loss3 = list(losses['loss3'])
    loss4 = list(losses['combined_losses'])

    y1 = loss4
    x = range(0,len(y1))
    plt.plot(x, y1, '.-')
    plt_title = file_str+'EPOCH: {} LR: {}'.format(EPOCH,LR)
    plt.title(plt_title)
    plt.xlabel('Iteration of All Epoch')
    plt.ylabel('LOSS1_'+file_str)
    plt.savefig(out_dir_img+'loss1_'+file_str)
    # import pdb;pdb.set_trace()



def init_weights_in_last_layers(net):
    torch.nn.init.constant_(net.ssd_d_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_e_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_f_block[6].weight, 1e-3)


##########################计算两个字符串的编辑距离#########################
# 三类错误，insert error,delete error,modification error
# dp算法
def Levenshtein_Distance(str1, str2):
    if type(str1) == float :
        # import pdb;pdb.set_trace()
        str1 = str(str1)
    if type(str2) == float :
        # import pdb;pdb.set_trace()
        str2 = str(str2)
    str1 = ''.join(filter(str.isalnum, str1)).upper()
    str2 = ''.join(filter(str.isalnum, str2)).upper()
    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    if str1 == '[padding]':
        str1 = ''
    if str2 == '[padding]':
        str2 = ''
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return 1 - matrix[len(str1)][len(str2)]/len(str2)


def plot_label_or_output(indices, str_file, count_now):
    chan_r = torch.eq(torch.gt(2 * tensor.new_ones(256, 128), indices), torch.gt(indices, tensor.new_zeros(256, 128)))
    chan_g = torch.gt(indices, 2 * tensor.new_ones(256, 128))
    chan_b = torch.eq(torch.gt(indices, tensor.new_zeros(256, 128)), torch.gt(4 * tensor.new_ones(256, 128), indices))

    output_trans = torch.stack((chan_r, chan_g, chan_b), 0)
    print(output_trans.cpu().size())
    pic = toPIL(output_trans.float().cpu())
    pic.save(os.path.join(out_dir_img, str_file +'_'+ count_now[0]).replace('npy', 'jpg'))
    print(str_file+".jpg saved")


# def compare_str(predicted_label, count_now,war_5):
def compare_str(predicted_label, count_now,war_5,class_5,num):
    dt = pd.read_csv(os.path.join(
        '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/pd_bertgrid_reduced_resized/',
        count_now[0]).replace('npy', 'csv'))

    test_dic = {1: '[padding]',
                2: '[padding]',
                3: '[padding]',
                4: '[padding]'
                }
    predicted_indices = predicted_label
    for i in range(len(predicted_indices)):
        if predicted_indices[i] != 0:
            if test_dic[predicted_indices[i].item()] == '[padding]':
                test_dic[predicted_indices[i].item()] = dt.loc[i, 'word_pieces']
            else:
                test_dic[predicted_indices[i].item()] += dt.loc[i, 'word_pieces']

    pd.DataFrame(test_dic, index=[0]).to_csv(os.path.join(out_dir_img, 'test_' + count_now[0]).replace('npy', 'csv'))


    with open(os.path.join(dir_classes, count_now[0]).replace("npy", "json")) as f:
        gt_dic = json.load(f)
    # import pdb;pdb.set_trace()
    war_5[0] += Levenshtein_Distance(test_dic[1], gt_dic['company'])
    war_5[1] += Levenshtein_Distance(test_dic[2], gt_dic['date'])
    war_5[2] += Levenshtein_Distance(test_dic[3], gt_dic['address'])
    war_5[3] += Levenshtein_Distance(test_dic[4], gt_dic['total'])
    
    class_5[num,0] = Levenshtein_Distance(test_dic[1], gt_dic['company'])
    class_5[num,1] = Levenshtein_Distance(test_dic[2], gt_dic['date'])
    class_5[num,2] = Levenshtein_Distance(test_dic[3], gt_dic['address'])
    class_5[num,3] = Levenshtein_Distance(test_dic[4], gt_dic['total'])

    return war_5

def conj_filter(predicted_indices, iter):
    for turn in range(iter):
        forward_filter_predicted_indices = torch.zeros((predicted_indices.shape[0]), dtype=int)
        forward_filter_predicted_indices_mask = torch.zeros((predicted_indices.shape[0]), dtype=int)
        backward_filter_predicted_indices = torch.zeros((predicted_indices.shape[0]), dtype=int)
        backward_filter_predicted_indices_mask = torch.zeros((predicted_indices.shape[0]), dtype=int)
        for j in range(gt_cord.shape[1]):
            if predicted_indices[j] != 0:
                if j + 2 < gt_cord.shape[1]:
                    if predicted_indices[j + 1] != predicted_indices[j] or predicted_indices[j + 2] != predicted_indices[j]:
                        # import pdb;pdb.set_trace()
                        forward_filter_predicted_indices[j] = 0
                    else:
                        forward_filter_predicted_indices[j] = predicted_indices[j]
                        forward_filter_predicted_indices_mask[j] = 1
            reverse_j = gt_cord.shape[1] - 1 - j
            if predicted_indices[reverse_j] != 0:
            # if predicted_indices[reverse_j] == 2:
                if reverse_j - 2 > 0:
                    if predicted_indices[reverse_j - 1] != predicted_indices[reverse_j] or predicted_indices[
                        reverse_j - 2] != predicted_indices[reverse_j]:
                        backward_filter_predicted_indices[reverse_j] = 0
                    else:
                        backward_filter_predicted_indices[reverse_j] = predicted_indices[reverse_j]
                        backward_filter_predicted_indices_mask[reverse_j] = 1

        predicted_indices = (forward_filter_predicted_indices_mask + ~(
            forward_filter_predicted_indices_mask.bool()) * backward_filter_predicted_indices_mask).to(
            device0) * predicted_indices
    return predicted_indices


if __name__ == '__main__':
    # in the name of reproducibility
    #device = torch.nn.DataParallel(torch, device_ids=[2, 3]).cuda()


    torch.manual_seed(0)

    # HW = 127-32+1
    HW = 768
    C = 64
    # C = 64
    num_classes = 5
    num_anchors = 4

    trainloader, testloader = new_ChargridDataset_test.get_dataset()

    net = new_ChargridNetwork(HW, C, num_classes, num_anchors)
    #net = nn.DataParallel(ChargridNetwork(3, 64, 5, 4),device_ids=[0,2,3])
    #net = net.apply(init_weights)

    # checkpoint = torch.load(os.path.join(model_dir, 'epoch-'+file_str+'-{}.pt'.format(EPOCH)), map_location=device0)
    checkpoint = torch.load(os.path.join(model_dir, 'epoch-'+file_str+'-{}.pt'.format(EPOCH)), map_location=device0)
    net.load_state_dict(checkpoint['model_state_dict'])
    #load_state_keywise(net, os.path.join(model_dir, 'epoch-{}.pt'.format(EPOCH)))
    plot_loss(checkpoint)
    # loss1 = FocalLoss2d()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(net.to(device0).parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(net.to(device0).parameters(), lr=LR, weight_decay=0.0001)
    optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])

    # loss1 = FocalLoss2d()
    loss4 = nn.CrossEntropyLoss(weight = log_weight)
    losses = {
        'loss1': [],
        'combined_losses': []
    }
    war_5 = torch.zeros(5)
    class_5 = torch.zeros(47,5)
    net.eval()
    for epoch in range(1):
        final_loss = 0.0
        #num_epo = epoch
        #print(num_epo)
        num = 0
        for inputs, label1, count_now in trainloader:
            num += 1
            print("im here")
            inputs = inputs.to(device0)
            label1 = label1.to(device0)
            # count_now = count_now.item()
            toPIL = transforms.ToPILImage()
            output1, output2, output3 = net(inputs.float())

            softmax = nn.Softmax(dim=1)
            output1_softmax =softmax(output1)


            # values, indice = torch.max(label1, dim=1)

####################################
            tensor = torch.tensor((), dtype=torch.int32)
            softmax = nn.Softmax(dim=1)
            # values , indice = torch.max(softmax(output1) , dim = 1)
            label_indices = torch.zeros((256,128))
            indices = torch.zeros((256,128))
            output_temp = output1[0, :, :, :]
            label_temp = label1[0, :, :, :].reshape(5, -1)
            gt_cord = label_temp[:4, :]
            gt_label = label_temp[4, :]
            gt_cord = gt_cord[:, :int(torch.nonzero(gt_label > -1, as_tuple=False)[-1] + 1)]
            gt_cord = gt_cord.int()
            gt_label = gt_label[:int(torch.nonzero(gt_label > -1, as_tuple=False)[-1] + 1)]
            gt_label = gt_label.int()
            predicted_label = torch.zeros((1,output1.shape[1], gt_cord.shape[1])).to(device0)

            for j in range(gt_cord.shape[1]):
                bbox = output_temp[:, gt_cord[2, j]:gt_cord[3, j], gt_cord[0, j]:gt_cord[1, j]]
                # output_bbox_pool = torch.nn.functional.avg_pool2d(bbox, kernel_size = (bbox.shape[1],bbox.shape[2])).reshape(5)
                output_bbox_pool = torch.nn.functional.max_pool2d(bbox,kernel_size=(bbox.shape[1], bbox.shape[2])).reshape(5)
                # predicted_label[:,:,j] = softmax(output_bbox_pool)
                predicted_label[:, :, j] = output_bbox_pool

                # indices[gt_cord[2, j]:gt_cord[3, j], gt_cord[0, j]:gt_cord[1, j]] =
                label_indices[gt_cord[2, j]:gt_cord[3, j], gt_cord[0, j]:gt_cord[1, j]] = gt_label[j]

            predicted_values, predicted_indices = torch.max(softmax(predicted_label.permute(2, 1, 0).view(gt_cord.shape[1], output1.shape[1])),dim=1)


            iter = 2
            # predicted_indices = conj_filter(predicted_indices, iter)
            print(loss4(predicted_label.permute(2, 1, 0).view(gt_cord.shape[1], output1.shape[1]), gt_label.long()))
            for j in range(gt_cord.shape[1]):
                indices[gt_cord[2, j]:gt_cord[3, j], gt_cord[0, j]:gt_cord[1, j]] = predicted_indices[j]

            print(indices.type())
            plot_label_or_output(indices, 'predicted1', count_now)
            plot_label_or_output(label_indices, 'label1', count_now)


            war_5 = compare_str(predicted_indices, count_now, war_5, class_5, num)
            # war_5 = compare_str(predicted_indices, count_now, war_5)
        import pdb;pdb.set_trace()
        war_5[4] = war_5.sum()/4
        war_5 /= 46
        import pdb;pdb.set_trace()









