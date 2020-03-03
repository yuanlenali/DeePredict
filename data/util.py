import os
import numpy
import shutil
import pickle
import random
from collections import Counter


def subsample_raw_data(data_size):
    if not os.path.isdir('subsampled_raw_data'):
        os.mkdir('subsampled_raw_data')
    print(os.getcwd())
    fin = open('raw_data/train.txt', 'r')
    fout = open('subsampled_raw_data/subsampled_train_' + str(data_size) + ".txt", 'w')
    for line_idx, line in enumerate(fin):
        fout.write(line)
        # do not subsample dataset if data_size < 0
        if data_size > 0 and line_idx == data_size - 1:
            fout.close()
            break

    fout.close()
    fin.close()


def split_data(data_size):
    split_rate = 0.9
    train_data_size = int(split_rate * data_size)

    fin = open('subsampled_raw_data/subsampled_train_' + str(data_size) + '.txt', 'r')
    fout_train = open('train_data/train.txt', 'w')
    fout_test = open('test_data/test.txt', 'w')

    idx = [i for i in range(data_size)]
    random.shuffle(idx)
    train_idx = idx[:train_data_size]
    
    for line_idx, line in enumerate(fin):
        if line_idx in train_idx:
            fout_train.write(line)
        else:
            fout_test.write(line)
    fin.close()
    fout_train.close()
    fout_test.close()


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # print('generate a feature dict')
        # Count the number of occurrences of sparse features
        feat_cnt = Counter()
        with open('train_data/train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])

        # Only retain sparse features with high frequency
        dis_feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                dis_feat_set.add(feat)

        # Create a dictionary for dense and sparse features
        feat_dict = {}
        tc = 1
        # dense features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        # sparse features
        cnt_feat_set = set()
        with open('train_data/train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '' or features[idx] not in dis_feat_set:
                        continue
                    if features[idx] not in cnt_feat_set:
                        cnt_feat_set.add(features[idx])
                        feat_dict[features[idx]] = tc
                        tc += 1

        # Save dictionary
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)
