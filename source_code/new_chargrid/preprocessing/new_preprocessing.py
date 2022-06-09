import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import torch


dir_filename = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/0325updated.task1train(626p)/'
outdir_np_img = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/np_img'
outdir_np_img_raw = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/np_img_raw'
outdir_np_chargrid = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/np_chargrid'
outdir_pd_chargrid = '/disk/lindonggraduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/pd_chargrid'




list_filenames = [f for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[1:12]))

gt_dic = torch.load('/disk/lindong/graduate_task/chargrid-pytorch-master/2021_train.pth')


def get_reduced_output(chargrid_pd, img_shape):
    chargrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))

    for index, row in chargrid_pd.iterrows():
        chargrid_np[row['top']:row['bot'], row['left']:row['right']] = int(row['ord'])

    plt.imshow(chargrid_np)
    plt.savefig(os.path.join(outdir_np_img_raw, filename).replace("txt", "png"))
    plt.close()
    
    gt_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))

    ## Remove empty rows and columns
    bool_y_empty = np.all(chargrid_np == 0, axis=0)
    bool_x_empty = np.all(chargrid_np == 0, axis=1)
    tab_cumsum_todelete_x = np.cumsum(bool_y_empty)
    tab_cumsum_todelete_y = np.cumsum(bool_x_empty)

    # import pdb;pdb.set_trace()

    chargrid_pd['left'] -= tab_cumsum_todelete_x[chargrid_pd['left'].tolist()]
    chargrid_pd['right'] -= tab_cumsum_todelete_x[chargrid_pd['right'].tolist()]

    chargrid_pd['bot'] -= tab_cumsum_todelete_y[chargrid_pd['bot'].tolist()]
    chargrid_pd['top'] -= tab_cumsum_todelete_y[chargrid_pd['top'].tolist()]

    x_shape = img_shape[0] - tab_cumsum_todelete_y[img_shape[0]-1]
    y_shape = img_shape[1] - tab_cumsum_todelete_x[img_shape[1]-1]

    chargrid_np = chargrid_np[:, ~bool_y_empty]
    chargrid_np = chargrid_np[~bool_x_empty, :]

    return chargrid_np, chargrid_pd


if __name__ == "__main__":
    num = 0
    max_lenth = 0
    id_max  = 0
    for filename in list_filenames:
        img_shape_str = gt_dic[filename][3]
        img_shape_str = img_shape_str.split(',')
        img_shape = [int(i) for i in img_shape_str]

        img_content_str_raw = gt_dic[filename][0]
        img_content_str = gt_dic[filename][0].split('\t')

        lenth_of_line = [len(i) for i in img_content_str]
        len_character = np.array(lenth_of_line).sum()


        img_label_np = gt_dic[filename][1]
        if (len(img_content_str_raw) != len(img_label_np)):
            raise ValueError('The number of raw img character {} dont equal to the number of label {}'.format(len(img_content_str_raw), len(img_label_np)))
        label_mask = np.ones(len(img_label_np),dtype=bool)
        for i in range(len(img_content_str_raw)):
            if img_content_str_raw[i] == '\t':
                label_mask[i] = False
        img_label_np = img_label_np[label_mask]
        if(len(img_label_np) != len_character):
            raise ValueError('The number of img character {} dont equal to the number of label {}'.format(len_character, len(img_label_np)))


        img_bbox_str = gt_dic[filename][2].split('\n')
        img_bbox_str = img_bbox_str[:-1]
        img_bbox_left,img_bbox_top,img_bbox_right,img_bbox_bot = [],[],[],[]
        for i in range(len(img_bbox_str)):
            temp_list_str = img_bbox_str[i].split(',')
            temp_list_int = [int(i) for i in temp_list_str]
            img_bbox_left.append(temp_list_int[0])
            img_bbox_top.append(temp_list_int[1])
            img_bbox_right.append(temp_list_int[2])
            img_bbox_bot.append(temp_list_int[3])
        if(len(img_bbox_str) != len(img_content_str)):
            raise ValueError('The number of img content lines {} dont equal to the number of bounding box {}'.format(len(img_content_str),len(img_bbox_str)))
        text_line_dict={
            'left':img_bbox_left,
            'top':img_bbox_top,
            'right':img_bbox_right,
            'bot':img_bbox_bot,
            'text':img_content_str
        }
        text_dt = pd.DataFrame(text_line_dict)
        # import pdb;pdb.set_trace()
        chargrid_pd = pd.DataFrame(columns=['left', 'top', 'right', 'bot', 'text','ord'])
        for index, row in text_dt.iterrows():
            char_width = (row['right'] - row['left'])//len(row['text'])
            for i in range(len(row['text'])):
                chargrid_pd = chargrid_pd.append({
                'left': row['left'] + i*char_width,
                'top': row['top'],
                'right': row['left'] + (i+1)*char_width,
                'bot': row['bot'],
                'text': row['text'][i],
                'ord': ord(row['text'][i])
                }, ignore_index=True)
        chargrid_pd['label'] = img_label_np

        chargrid_np, chargrid_pd = get_reduced_output(chargrid_pd, img_shape)

        np.save(os.path.join(outdir_np_chargrid, filename).replace('txt','npy'),chargrid_np)
        chargrid_pd.to_csv(os.path.join(outdir_pd_chargrid, filename).replace('txt','csv'))

        plt.imshow(chargrid_np)
        plt.savefig(os.path.join(outdir_np_img, filename).replace("txt", "png"))
        plt.close()

        print(num,',done!')
        num += 1


    import pdb;pdb.set_trace()
