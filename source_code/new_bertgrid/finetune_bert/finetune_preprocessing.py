import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

dir_np_bertgrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/np_bertgrid_reduced_resized'
dir_pd_bertgrid_reduced_resized = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/preprocessing_ter/pd_bertgrid_reduced_resized'
outdir_input_ids = '/disk/lindong/graduate_task/chargrid-pytorch-master/chargrid-pytorch-master/data/new_Bert/finetune_preprocessing/input_ids'
dir_filename = dir_np_bertgrid_reduced_resized
list_filenames = [f.replace('npy','csv') for f in os.listdir(dir_filename) if os.path.isfile(os.path.join(dir_filename, f))]
list_filenames.sort(key = lambda x: int(x[0:3]))

max_lenth = 512

if __name__ == "__main__":
    ## Load inputs
    num = 0
    for filename in list_filenames:
        pd_bertgrid = pd.read_csv(os.path.join(dir_pd_bertgrid_reduced_resized,filename))
        input_ids_list = pd_bertgrid['input_ids'].tolist()
        input_ids_list.insert(0,101)
        input_ids_list.append(102)
        if len(input_ids_list) > max_lenth:
            raise ValueError('Input ID is too long {}'.format(len(input_ids_list)))
        add_list = (-1*np.ones((max_lenth-len(input_ids_list)))).astype(int).tolist()
        final_input_ids_list = input_ids_list + add_list
        final_input_ids_np = np.array(final_input_ids_list)
        np.save(os.path.join(outdir_input_ids,filename).replace('csv','npy'),final_input_ids_np)
        print(num,',done!')
        num += 1
    import pdb;pdb.set_trace()