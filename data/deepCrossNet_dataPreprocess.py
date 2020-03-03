from util import *


if __name__ == '__main__':
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    data_size = 50000
    if os.path.exists('subsampled_raw_data/subsampled_train_' + str(data_size) + ".txt"):
        os.remove('subsampled_raw_data/subsampled_train_' + str(data_size) + ".txt") 
    if os.path.exists('train_data/train.txt'):
        os.remove('train_data/train.txt') 
    if os.path.exists('test_data/test.txt'):
        os.remove('test_data/test.txt') 

    subsample_raw_data(data_size)    
    split_data(data_size)
    get_feat_dict()
    print('Done!')