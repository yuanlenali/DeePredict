import os
import numpy
import shutil
import pickle
from collections import Counter


def subsample_raw_data(data_size):
    if not os.path.isdir('subsampled_raw_data'):
        os.mkdir('subsampled_raw_data')
    print(os.getcwd())
    fin = open('raw_data/train.txt', 'r')
    fout = open('subsampled_raw_data/subsampled_train_' + str(data_size) + ".txt", 'w')
    for line_idx, line in enumerate(fin):
        fout.write(line)
        if line_idx == data_size - 1:
            fout.close()
            break

    fout.close()
    fin.close()