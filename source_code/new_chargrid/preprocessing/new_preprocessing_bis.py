import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

## Hyperparameters
dir_np_chargrid = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/np_chargrid'
dir_pd_chargrid = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing/pd_chargrid'

outdir_np_img_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/np_img_reduced'
outdir_np_chargrid_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/np_chargrid_reduced'
outdir_pd_chargrid_reduced = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/CHAR/preprocessing_bis/pd_chargrid_reduced'


equal_threshold = 0.95
max_padding = 3
dir_filename = dir_np_chargrid


list_filenames = [f for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[1:12]))

def get_reduce(img, axis):
    reduce_f = 1

    trust = 1.0
    reduce = 1
    while reduce <= img.shape[axis] / 2:
        reduce += 1
        if img.shape[axis] % reduce == 0:
            if axis == 0:
                img_reshaped = img.reshape(img.shape[0] // reduce, -1, img.shape[1])
                img2 = np.repeat(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=img_reshaped),
                                 reduce, axis=axis)
            else:
                img_reshaped = img.reshape(img.shape[0], img.shape[1] // reduce, -1)
                img2 = np.repeat(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=img_reshaped),
                                 reduce, axis=axis)

            trust = np.sum(img == img2) / (np.shape(img)[0] * np.shape(img)[1])
            if trust > equal_threshold:
                reduce_f = reduce

    return reduce_f


def get_max_reduce(img, axis):
    reduce_f = get_reduce(img, axis)
    padding_left = 0
    padding_right = 0

    for i in range(0, max_padding):
        img = np.insert(img, 0, 0, axis=axis)
        reduce_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f = reduce_f_
            padding_left = i + 1
            padding_right = i

        img = np.insert(img, 0, img.shape[axis], axis=axis)
        reduce_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f = reduce_f_
            padding_left = i + 1
            padding_right = i + 1

    return reduce_f, padding_left, padding_right


def get_img_reduced(img, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot):
    img2 = img
    for i in range(0, padding_top):
        img2 = np.insert(img2, 0, 0, axis=0)
    for i in range(0, padding_bot):
        img2 = np.insert(img2, 0, img2.shape[0], axis=0)
    for i in range(0, padding_left):
        img2 = np.insert(img2, 0, 0, axis=1)
    for i in range(0, padding_right):
        img2 = np.insert(img2, 0, img2.shape[1], axis=1)

    import pdb;pdb.set_trace()
    img2_reshaped = img2.reshape(img2.shape[0] // reduce_y, -1, img2.shape[1])
    img2 = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=img2_reshaped)

    img2_reshaped = img2.reshape(img2.shape[0], img2.shape[1] // reduce_x, -1)
    img2 = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=img2_reshaped)

    return img2


def reduce_pd_bbox(pd_bbox, padding_left, padding_top, reduce_x, reduce_y):
    # import pdb;pdb.set_trace()
    pd_bbox['left'] += padding_left
    pd_bbox['right'] += padding_left
    pd_bbox['top'] += padding_top
    pd_bbox['bot'] += padding_top

    pd_bbox['left'] = pd_bbox['left'].astype(float)
    pd_bbox['right'] = pd_bbox['right'].astype(float)
    pd_bbox['top'] = pd_bbox['top'].astype(float)
    pd_bbox['bot'] = pd_bbox['bot'].astype(float)

    pd_bbox['left'] = round(pd_bbox['left'] / reduce_x)
    pd_bbox['right'] = round(pd_bbox['right'] / reduce_x)
    pd_bbox['top'] = round(pd_bbox['top'] / reduce_y)
    pd_bbox['bot'] = round(pd_bbox['bot'] / reduce_y)

    pd_bbox['left'] = pd_bbox['left'].astype(int)
    pd_bbox['right'] = pd_bbox['right'].astype(int)
    pd_bbox['top'] = pd_bbox['top'].astype(int)
    pd_bbox['bot'] = pd_bbox['bot'].astype(int)

    return pd_bbox

if __name__ == "__main__":
    num = 0
    for filename in list_filenames:
        print(filename)
        ## Load inputs
        np_chargrid = np.load(os.path.join(dir_np_chargrid, filename), allow_pickle=True)
        pd_chargrid = pd.read_csv(os.path.join(dir_pd_chargrid,filename).replace('npy','csv'))

        if np.shape(np_chargrid) != (0, 0):
            reduce_y, padding_top, padding_bot = get_max_reduce(np_chargrid, 0)
            print("final reduce_y = ", reduce_y, "padding_t = ", padding_top, "padding_b = ", padding_bot, filename)
            #
            reduce_x, padding_left, padding_right = get_max_reduce(np_chargrid, 1)
            print("final reduce_x = ", reduce_x, "padding_l = ", padding_left, "padding_r = ", padding_right, filename)

            np_chargrid_reduced = get_img_reduced(np_chargrid, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot)

            pd_chargrid_reduced = reduce_pd_bbox(pd_chargrid, padding_left, padding_top, reduce_x, reduce_y)



            np.save(os.path.join(outdir_np_chargrid_reduced, filename).replace('npy', 'npy'), np_chargrid_reduced)
            pd_chargrid_reduced.to_csv(os.path.join(outdir_pd_chargrid_reduced, filename).replace('npy', 'csv'))

            plt.imshow(np_chargrid_reduced)
            plt.savefig(os.path.join(outdir_np_img_reduced, filename).replace("npy", "png"))
            plt.close()
            # import pdb;pdb.set_trace()
            print(num,filename+',done!')
            num += 1
