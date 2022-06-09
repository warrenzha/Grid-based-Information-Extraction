import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.transform import resize


## Hyperparameters
dir_np_img_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/np_img_reduced'
dir_np_chargrid_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/np_chargrid_reduced'
dir_pd_chargrid_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/pd_chargrid_reduced'

outdir_np_chargrid_cord = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/np_chargrid_cord'
outdir_np_chargrid_reduced_resized_1h = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/np_chargrid_reduced_resized_1h'
outdir_pd_chargrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/pd_chargrid_reduced_resized'
outdir_img_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_ter/img_reduced_resized'

dir_filename = dir_np_chargrid_reduced
list_filenames = [f for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[1:12]))

target_height = 256
target_width = 128
target_digit = 127-32+1
target_class = 5
nb_anchors = 4  # one per foreground class

#最长的图片的字符个数为549
max_lenth = 1273

def convert_to_1h(img):
    img_1h = np.eye(target_digit)[img]
    return img_1h


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

    np_chargrid_reduced_resized = np.zeros((target_height,target_width))
    for index,row in pd_bbox.iterrows():
        np_chargrid_reduced_resized[row['top']:row['bot'], row['left']:row['right']] = row['ord'] - 31
        # np_chargrid_reduced_resized[row['top']:row['bot'], row['left']:row['right']] = row['ord']
    # import pdb;pdb.set_trace()
    np_chargrid_reduced_resized = np_chargrid_reduced_resized.astype(int)
    np_chargrid_cord = np.zeros((5,  max_lenth))
    if max_lenth < len(pd_bbox.index.values):
        raise ValueError('The lenth of pd_bbox is longer than max_lenth'.format(pd_bbox.index.values))
    # import pdb;pdb.set_trace()
    np_chargrid_cord[0, :len(pd_bbox.index.values)] = np.array(pd_bbox['left'].tolist())
    np_chargrid_cord[1, :len(pd_bbox.index.values)] = np.array(pd_bbox['right'].tolist())
    np_chargrid_cord[2, :len(pd_bbox.index.values)] = np.array(pd_bbox['top'].tolist())
    np_chargrid_cord[3, :len(pd_bbox.index.values)] = np.array(pd_bbox['bot'].tolist())
    np_chargrid_cord[4, :len(pd_bbox.index.values)] = np.array(pd_bbox['label'].tolist())
    if len(pd_bbox.index.values) != max_lenth:
        np_chargrid_cord[0, len(pd_bbox.index.values): ] = -1
        np_chargrid_cord[1, len(pd_bbox.index.values): ] = -1
        np_chargrid_cord[2, len(pd_bbox.index.values): ] = -1
        np_chargrid_cord[3, len(pd_bbox.index.values): ] = -1
        np_chargrid_cord[4, len(pd_bbox.index.values): ] = -1
    # import pdb;pdb.set_trace()
    return np_chargrid_reduced_resized, np_chargrid_cord

def extract_anchor_coordinates_full(pd_bbox_full, img_shape):
    # import pdb;pdb.set_trace()
    pd_bbox_full['top_left_x'] /= img_shape[1]
    pd_bbox_full['bot_left_x'] /= img_shape[1]
    pd_bbox_full['top_right_x'] /= img_shape[1]
    pd_bbox_full['bot_right_x'] /= img_shape[1]
    pd_bbox_full['top_left_y'] /= img_shape[0]
    pd_bbox_full['bot_left_y'] /= img_shape[0]
    pd_bbox_full['top_right_y'] /= img_shape[0]
    pd_bbox_full['bot_right_y'] /= img_shape[0]

    pd_bbox_full['top_left_x'] = (pd_bbox_full['top_left_x'] * target_width + 0.5).astype(int)
    pd_bbox_full['bot_left_x'] = (pd_bbox_full['bot_left_x'] * target_width + 0.5).astype(int)
    pd_bbox_full['top_right_x'] = (pd_bbox_full['top_right_x'] * target_width + 0.5).astype(int)
    pd_bbox_full['bot_right_x'] = (pd_bbox_full['bot_right_x'] * target_width + 0.5).astype(int)
    pd_bbox_full['top_left_y'] = (pd_bbox_full['top_left_y'] * target_height + 0.5).astype(int)
    pd_bbox_full['bot_left_y'] = (pd_bbox_full['bot_left_y'] * target_height + 0.5).astype(int)
    pd_bbox_full['top_right_y'] = (pd_bbox_full['top_right_y'] * target_height + 0.5).astype(int)
    pd_bbox_full['bot_right_y'] = (pd_bbox_full['bot_right_y'] * target_height + 0.5).astype(int)


if __name__ == "__main__":

    ## Load inputs
    tab_np_img = []
    tab_gt = []
    for i in range(0, len(list_filenames)):
        tab_np_img.append(np.load(os.path.join(dir_np_chargrid_reduced, list_filenames[i])))

    print("tab_img shape=", np.shape(tab_np_img))
    num = 0
    for i in range(0, len(list_filenames)):

        ## Load input
        pd_chargrid_reduced = pd.read_csv(os.path.join(dir_pd_chargrid_reduced, list_filenames[i]).replace("npy", "csv"))
        np_chargrid_reduced_resized, np_chargrid_cord = extract_anchor_coordinates(pd_chargrid_reduced, np.shape(tab_np_img[i]))

        plt.imshow(np_chargrid_reduced_resized)
        plt.savefig(os.path.join(outdir_img_reduced_resized, list_filenames[i]).replace("npy", "png"))
        plt.close()

        np_chargrid_reduced_resized_1h = convert_to_1h(np_chargrid_reduced_resized)

        np.save(os.path.join(outdir_np_chargrid_reduced_resized_1h,list_filenames[i]),np_chargrid_reduced_resized_1h)
        np.save(os.path.join(outdir_np_chargrid_cord, list_filenames[i]),np_chargrid_cord)
        pd_chargrid_reduced.to_csv(os.path.join(outdir_pd_chargrid_reduced_resized, list_filenames[i]).replace('npy','csv'))
        print(num,list_filenames[i]+', done!')
        num += 1
        # import pdb;pdb.set_trace()