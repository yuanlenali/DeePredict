from util import *


if __name__ == '__main__':
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    data_size = 50000
    subsample_raw_data(data_size)
    split_data(data_size)
    get_feat_dict()
    print('Done!')
