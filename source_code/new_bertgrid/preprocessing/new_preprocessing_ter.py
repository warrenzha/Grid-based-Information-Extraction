import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import AutoTokenizer


## Hyperparameters
dir_np_img_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_bis/np_img_reduced'
dir_np_chargrid_reduced ='/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_bis/np_bertgrid_reduced/done'
dir_pd_chargrid_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_bis/pd_bertgrid_reduced'
dir_pd_label = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing/pd_label'

outdir_np_chargrid_cord = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/np_bertgrid_cord'
outdir_np_chargrid_reduced_resized_1h = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/np_bertgrid_reduced_resized'
outdir_pd_chargrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/pd_bertgrid_reduced_resized'
outdir_img_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/img_reduced_resized'

dir_filename = dir_np_chargrid_reduced
list_filenames = [f for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[1:12]))

target_height = 256
target_width = 128
target_digit = 768
target_class = 5
nb_anchors = 4  # one per foreground class

max_lenth = 510

tokenizer = AutoTokenizer.from_pretrained(
    r'/home/jinac/anaconda3/envs/zn/lib/python3.7/site-packages/pytorch_pretrained_bert/bert-base-uncased')
bert = BertModel.from_pretrained(
    r'/home/jinac/anaconda3/envs/zn/lib/python3.7/site-packages/pytorch_pretrained_bert/bert-base-uncased')

# tokenizer = AutoTokenizer.from_pretrained(
#     r'D:\deep_learning\anaconda\envs\pytorch\Lib\site-packages\pytorch_pretrained_bert\bert-base-uncased')
# bert = BertModel.from_pretrained(
#     r'D:\deep_learning\anaconda\envs\pytorch\Lib\site-packages\pytorch_pretrained_bert\bert-base-uncased')


def extract_anchor_coordinates(pd_bbox, img_shape):
    # import pdb;pdb.set_trace()
    pd_bbox['left'] /= img_shape[1]
    pd_bbox['right'] /= img_shape[1]
    pd_bbox['top'] /= img_shape[0]
    pd_bbox['bot'] /= img_shape[0]

    pd_bbox['left'] = (pd_bbox['left'] * target_width + 0.5).astype(int)
    pd_bbox['right'] = (pd_bbox['right'] * target_width + 0.5).astype(int)
    pd_bbox['top'] = (pd_bbox['top'] * target_height + 0.5).astype(int)
    pd_bbox['bot'] = (pd_bbox['bot'] * target_height + 0.5).astype(int)

    # import pdb;pdb.set_trace()
    bertgrid_pd = get_bert_pd(pd_bbox)
    np_bertgrid = np.zeros((target_height,target_width,768))
    img = np.zeros((target_height,target_width))
    if len(bertgrid_pd.index.values)>510:
        raise ValueError('input is too long ,the lenth of input is {}'.format(len(bertgrid_pd.index.values)))
    else:
        input_ids_all = bertgrid_pd['input_ids'].values.tolist()
        input_ids_all.insert(0, 101)
        input_ids_all.append(102)
        input_ids_all = torch.Tensor(input_ids_all).view(1, -1).long()
    # print(input_ids_all)
    result = bert(input_ids_all, output_all_encoded_layers=True)
    temp_list = []
    for index, row in bertgrid_pd.iterrows():
        np_bertgrid[row['top']:row['bot'], row['left']:row['right'], :] = (
            result[0][11].view(result[0][11].shape[1], result[0][11].shape[2])[index + 1]).detach().numpy()
        temp_list += row['word_pieces_label']
        img[row['top']:row['bot'], row['left']:row['right']] = index+10
    np_bertgrid_cord = np.zeros((5,  max_lenth))
    if max_lenth < len(bertgrid_pd.index.values):
        raise ValueError('The lenth of pd_bbox is longer than max_lenth'.format(pd_bbox.index.values))
    # import pdb;pdb.set_trace()
    np_bertgrid_cord[0, :len(bertgrid_pd.index.values)] = np.array(bertgrid_pd['left'].tolist())
    np_bertgrid_cord[1, :len(bertgrid_pd.index.values)] = np.array(bertgrid_pd['right'].tolist())
    np_bertgrid_cord[2, :len(bertgrid_pd.index.values)] = np.array(bertgrid_pd['top'].tolist())
    np_bertgrid_cord[3, :len(bertgrid_pd.index.values)] = np.array(bertgrid_pd['bot'].tolist())
    np_bertgrid_cord[4, :len(bertgrid_pd.index.values)] = np.array(temp_list)
    if len(pd_bbox.index.values) != max_lenth:
        np_bertgrid_cord[0, len(bertgrid_pd.index.values): ] = -1
        np_bertgrid_cord[1, len(bertgrid_pd.index.values): ] = -1
        np_bertgrid_cord[2, len(bertgrid_pd.index.values): ] = -1
        np_bertgrid_cord[3, len(bertgrid_pd.index.values): ] = -1
        np_bertgrid_cord[4, len(bertgrid_pd.index.values): ] = -1

    plt.imshow(img)
    plt.savefig(os.path.join(outdir_img_reduced_resized, list_filenames[i]).replace("npy", "png"))
    plt.close()
    return np_bertgrid, bertgrid_pd, np_bertgrid_cord
    # return np_chargrid_reduced_resized, np_chargrid_cord

def get_bert_pd(wordgrid_pd):
    word_pieces_pd = pd.DataFrame(columns=['left', 'top', 'right', 'bot', 'len_character', 'word_pieces', 'input_ids', 'label'])
    serialized_text = []
    serialized_word_pieces = []
    # import pdb;pdb.set_trace()
    for index, row in wordgrid_pd.iterrows():
        # print(row['text'])
        # if index == 100:
            # import pdb;pdb.set_trace()
        serialized_text.extend(row['text'].split("\n"))
        encoded_input = tokenizer(row['text'])
        # print(encoded_input)
        word_pieces_len = 0

        list_str_label = row['label'].lstrip('[').rstrip(']').split(' ')
        list_int_label = [int(i) for i in list_str_label]
        # import pdb;pdb.set_trace()
        # if len(set(list_int_label)) != 1:
        #     raise ValueError('The value of label in this word {} is not unique'.format(len(set(list_int_label))))
        word_label = set(list_int_label)

        for i in range(1, np.array(encoded_input['input_ids']).shape[0] - 1):
            str_word_pieces = tokenizer.decode(encoded_input['input_ids'][i]).split("\n")
            str_word_pieces = str_word_pieces[0].lstrip('#')
            if str_word_pieces == '':
                str_word_pieces = '#'
            label = list_int_label[word_pieces_len : word_pieces_len + len(str_word_pieces)]
            if len(set(label)) == 1:
                word_pieces_label = set(label)
            else:
                # import pdb;pdb.set_trace()
                raise ValueError('The value of label in this word piece {} is not unique'.format(len(set(label))))
            serialized_word_pieces.extend(str_word_pieces.split("\n"))
            word_pieces_pd = word_pieces_pd.append({
                'left': row['left'] + (row['right'] - row['left']) * word_pieces_len // len(row["text"]),
                'top': row['top'],
                'right': row['left'] + (row['right'] - row['left']) * word_pieces_len // len(row["text"])+
                         (row['right'] - row['left']) * len(str_word_pieces) // len(row["text"]),
                'bot': row['bot'],
                'len_character': len(str_word_pieces),
                'word_pieces': str_word_pieces,
                'input_ids': encoded_input['input_ids'][i],
                'label': label,
                'word_pieces_label': list(word_pieces_label)
            }, ignore_index=True)
            word_pieces_len += len(str_word_pieces)

    # import pdb;pdb.set_trace()
    return word_pieces_pd

if __name__ == "__main__":

    ## Load inputs
    tab_np_img = []
    tab_gt = []
    for i in range(0, len(list_filenames)):
        tab_np_img.append(np.load(os.path.join(dir_np_chargrid_reduced, list_filenames[i])))

    print("tab_img shape=", np.shape(tab_np_img))
    num = 0

    for i in range(0, len(list_filenames)):
        # import pdb;pdb.set_trace()
        pd_chargrid_reduced = pd.read_csv(os.path.join(dir_pd_chargrid_reduced, list_filenames[i]).replace("npy", "csv"))
        pd_label = pd.read_csv(os.path.join(dir_pd_label, list_filenames[i]).replace('npy', 'csv'))

        # import pdb;pdb.set_trace()
        pd_chargrid_reduced['label'] = pd_label['label']
        # import pdb;pdb.set_trace()
        np_bertgrid_reduced_resized, bertgrid_pd, np_bertgrid_cord = extract_anchor_coordinates(pd_chargrid_reduced, np.shape(tab_np_img[i]))
        # np_chargrid_reduced_resized_1h = convert_to_1h(np_chargrid_reduced_resized)
        # import pdb;pdb.set_trace()
        np.save(os.path.join(outdir_np_chargrid_reduced_resized_1h,list_filenames[i]),np_bertgrid_reduced_resized)
        np.save(os.path.join(outdir_np_chargrid_cord,list_filenames[i]),np_bertgrid_cord)
        bertgrid_pd.to_csv(os.path.join(outdir_pd_chargrid_reduced_resized, list_filenames[i]).replace('npy','csv'))
        if len(bertgrid_pd.index.values) > max_lenth:
            max_lenth = len(bertgrid_pd.index.values)

        print(num,list_filenames[i]+', done!')
        num += 1
    import pdb;pdb.set_trace()