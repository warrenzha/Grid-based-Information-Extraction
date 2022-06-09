import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import torch


dir_filename = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/0325updated.task1train(626p)'
dir_img = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/img'

outdir_np_img = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/np_img'
outdir_np_img_raw = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/np_img_raw'
outdir_np_bertgrid = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/np_bertgrid'
outdir_pd_bertgrid = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/pd_bertgrid'
outdir_pd_label = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/pd_label'
outdir_img_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/img_reduced_raw'

list_filenames = [f for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[1:12]))


gt_dic = torch.load('/disk/lindong/graduate_task/chargrid-pytorch-master/2021_train.pth')

def get_chargrid_output(filename):
    img_shape_str = gt_dic[filename][3]
    img_shape_str = img_shape_str.split(',')
    img_shape = [int(i) for i in img_shape_str]

    img_content_str_raw = gt_dic[filename][0]
    img_content_str = gt_dic[filename][0].split('\t')
    lenth_of_line = [len(i) for i in img_content_str]
    len_character = np.array(lenth_of_line).sum()
    img_label_np = gt_dic[filename][1]
    if (len(img_content_str_raw) != len(img_label_np)):
        raise ValueError(
            'The number of raw img character {} dont equal to the number of label {}'.format(len(img_content_str_raw),len(img_label_np)))
    label_mask = np.ones(len(img_label_np), dtype=bool)
    for i in range(len(img_content_str_raw)):
        if img_content_str_raw[i] == '\t':
            label_mask[i] = False
    img_label_np = img_label_np[label_mask]
    if (len(img_label_np) != len_character):
        raise ValueError('The number of img character {} dont equal to the number of label {}'.format(len_character, len( img_label_np)))

    img_bbox_str = gt_dic[filename][2].split('\n')
    img_bbox_str = img_bbox_str[:-1]
    img_bbox_left, img_bbox_top, img_bbox_right, img_bbox_bot = [], [], [], []
    for i in range(len(img_bbox_str)):
        temp_list_str = img_bbox_str[i].split(',')
        temp_list_int = [int(i) for i in temp_list_str]
        img_bbox_left.append(temp_list_int[0])
        img_bbox_top.append(temp_list_int[1])
        img_bbox_right.append(temp_list_int[2])
        img_bbox_bot.append(temp_list_int[3])
    if (len(img_bbox_str) != len(img_content_str)):
        raise ValueError('The number of img content lines {} dont equal to the number of bounding box {}'.format(
            len(img_content_str), len(img_bbox_str)))
    text_line_dict = {
        'left': img_bbox_left,
        'top': img_bbox_top,
        'right': img_bbox_right,
        'bot': img_bbox_bot,
        'text': img_content_str
    }
    text_dt = pd.DataFrame(text_line_dict)
    # import pdb;pdb.set_trace()
    chargrid_pd = pd.DataFrame(columns=['left', 'top', 'right', 'bot', 'text', 'ord'])
    for index, row in text_dt.iterrows():
        char_width = (row['right'] - row['left']) // len(row['text'])
        for i in range(len(row['text'])):
            chargrid_pd = chargrid_pd.append({
                'left': row['left'] + i * char_width,
                'top': row['top'],
                'right': row['left'] + (i + 1) * char_width,
                'bot': row['bot'],
                'text': row['text'][i],
                'ord': ord(row['text'][i])
            }, ignore_index=True)
    chargrid_pd['label'] = img_label_np
    return chargrid_pd

def get_reduced_output(chargrid_pd, text_dt, img_shape):
    chargrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))

    for index, row in chargrid_pd.iterrows():
        chargrid_np[row['top']:row['bot'], row['left']:row['right']] = int(row['ord'])

    bertgrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape((img_shape[0], img_shape[1]))
    ##################################
    for index, row in text_dt.iterrows():
        bertgrid_np[row['top']:row['bot'], row['left']:row['right']] = index+10
    # import pdb;pdb.set_trace()

    plt.imshow(bertgrid_np)
    plt.savefig(os.path.join(outdir_np_img_raw, filename).replace("txt", "png"))
    plt.close()

    dir_path = os.path.join(dir_img, filename).replace('txt','jpg')
    img = plt.imread(dir_path, format='jpeg')
    # import pdb;pdb.set_trace()
    ## Remove empty rows and columns
    bool_y_empty = np.all(chargrid_np == 0, axis=0)
    bool_x_empty = np.all(chargrid_np == 0, axis=1)
    tab_cumsum_todelete_x = np.cumsum(bool_y_empty)
    tab_cumsum_todelete_y = np.cumsum(bool_x_empty)


    chargrid_pd['left'] -= tab_cumsum_todelete_x[chargrid_pd['left'].tolist()]
    chargrid_pd['right'] -= tab_cumsum_todelete_x[chargrid_pd['right'].tolist()]

    chargrid_pd['bot'] -= tab_cumsum_todelete_y[chargrid_pd['bot'].tolist()]
    chargrid_pd['top'] -= tab_cumsum_todelete_y[chargrid_pd['top'].tolist()]

    text_dt['left'] -= tab_cumsum_todelete_x[text_dt['left'].tolist()]
    text_dt['right'] -= tab_cumsum_todelete_x[text_dt['right'].tolist()]

    text_dt['bot'] -= tab_cumsum_todelete_y[text_dt['bot'].tolist()]
    text_dt['top'] -= tab_cumsum_todelete_y[text_dt['top'].tolist()]

    x_shape = img_shape[0] - tab_cumsum_todelete_y[img_shape[0]-1]
    y_shape = img_shape[1] - tab_cumsum_todelete_x[img_shape[1]-1]

    bertgrid_np = bertgrid_np[:, ~bool_y_empty]
    bertgrid_np = bertgrid_np[~bool_x_empty, :]
    chargrid_np = chargrid_np[:, ~bool_y_empty]
    chargrid_np = chargrid_np[~bool_x_empty, :]

    img = img[:,~bool_y_empty,:]
    img = img[~bool_x_empty,:,:]
    plt.imshow(img)
    plt.savefig(os.path.join(outdir_img_reduced, filename).replace("txt", "jpg"))
    plt.close()

    np.save(os.path.join(outdir_np_img_raw,filename).replace('txt','npy'),img)
    # import pdb;pdb.set_trace()

    return bertgrid_np, text_dt


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

        ##############################################
        img_content_str_word = []
        len_list_line_to_word = []
        for i in range(len(img_content_str)):
            temp_content_str = img_content_str[i].split(' ')
            len_list_line_to_word.append(len(temp_content_str))
            img_content_str_word += temp_content_str


        lenth_of_line_word = [len(i) for i in img_content_str_word]
        len_character = np.array(lenth_of_line_word).sum()
        lenth_of_line_full = [len(i) for i in img_content_str]
        lenth_of_line = [lenth_of_line_full[i]-len_list_line_to_word[i]+1 for i in range(len(lenth_of_line_full))]
        lenth_of_line_word_char = []
        #
        lenth_of_line_word_char_corre = []
        for i in range(len(img_content_str)):
            np.repeat(lenth_of_line[i], len_list_line_to_word[i])
            lenth_of_line_word_char += np.repeat(lenth_of_line[i], len_list_line_to_word[i]).tolist()


        # import pdb;pdb.set_trace()
        img_label_np = gt_dic[filename][1]
        if (len(img_content_str_raw) != len(img_label_np)):
            raise ValueError('The number of raw img character {} dont equal to the number of label {}'.format(len(img_content_str_raw), len(img_label_np)))
        label_mask = np.ones(len(img_label_np),dtype=bool)
        # import pdb;pdb.set_trace()
        for i in range(len(img_content_str_raw)):
            if img_content_str_raw[i] == '\t':
                label_mask[i] = False
            if img_content_str_raw[i] == ' ':
                label_mask[i] = False
        img_label_np = img_label_np[label_mask]
        if(len(img_label_np) != len_character):
            raise ValueError('The number of img character {} dont equal to the number of label {}'.format(len_character, len(img_label_np)))
        img_bbox_str = gt_dic[filename][2].split('\n')
        img_bbox_str = img_bbox_str[:-1]
        img_bbox_left,img_bbox_top,img_bbox_right,img_bbox_bot = [],[],[],[]
        count = 0
        pos_count = 0
        label_word = []
        for i in range(len(img_bbox_str)):
            temp_list_str = img_bbox_str[i].split(',')
            temp_list_int = [int(i) for i in temp_list_str]
            top_left = temp_list_int[0]

            for j in range(len_list_line_to_word[i]):
                # import pdb;pdb.set_trace()
                word_width = int((temp_list_int[2] - temp_list_int[0])*(lenth_of_line_word[count]/lenth_of_line_word_char[count]))

                label_word.append(img_label_np[pos_count : pos_count+lenth_of_line_word[count]])

                img_bbox_left.append(top_left)
                img_bbox_top.append(temp_list_int[1])
                img_bbox_right.append(top_left+word_width)
                img_bbox_bot.append(temp_list_int[3])
                top_left += word_width
                pos_count += lenth_of_line_word[count]
                count += 1


        # import pdb;pdb.set_trace()
        label_word_dict = {'label': label_word}
        label_dt = pd.DataFrame(label_word_dict)

        label_dt.to_csv(os.path.join(outdir_pd_label, filename).replace('txt', 'csv'))
        if(len(img_bbox_str) != len(img_content_str)):
            raise ValueError('The number of img content lines {} dont equal to the number of bounding box {}'.format(len(img_content_str),len(img_bbox_str)))
        #建立dataframe文件
        text_line_dict={
            'left':img_bbox_left,
            'top':img_bbox_top,
            'right':img_bbox_right,
            'bot':img_bbox_bot,
            'text':img_content_str_word,
            'len_chracter':lenth_of_line_word
        }
        text_dt = pd.DataFrame(text_line_dict)
        # import pdb;pdb.set_trace()

        chargrid_pd = get_chargrid_output(filename)
        bertgrid_np, bertgrid_pd = get_reduced_output(chargrid_pd, text_dt, img_shape)

        # np.save(os.path.join(outdir_np_bertgrid, filename).replace('txt','npy'),bertgrid_np)
        bertgrid_pd.to_csv(os.path.join(outdir_pd_bertgrid, filename).replace('txt','csv'))

        # plt.imshow(bertgrid_np)
        # plt.savefig(os.path.join(outdir_np_img, filename).replace("txt", "png"))
        # plt.close()

        print(num,',done!')
        num += 1

    import pdb;pdb.set_trace()
