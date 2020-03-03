import re
import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EPOCHS = 10
BATCH_SIZE = 2048


def train_test_model(model, device, train_data_path, test_data_path, feat_dict_):
    """
    train and test the model # of epochs
    :param model: The NN based model
    :param train_data_path: where the train data locates
    :param train_data_path: where the test data locates
    :param feat_dict_: a feature dict, e.g. {feature_name: idx} 
    :return:
    """
    print("Start Training Model!")

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_data_path, feat_dict_, device, optimizer, epoch)
        test_model(model, test_data_path, feat_dict_, device)


""" ************************************************************************************ """
"""                      Using Criteo DataSet to train/test Model                        """
""" ************************************************************************************ """
def train_model(model, train_data_path, feat_dict_, device, optimizer, epoch,
                use_reg_l1=False, use_reg_l2=False):
    """
    train one epoch of the model batch by batch
    :param model: The NN based model
    :param train_data_path: where the train data locates
    :param feat_dict_: a feature dict, e.g. {feature_name: idx} 
    :param optimizer: use optimizer for update weights
    :param epoch: the current epoch
    :return:
    """
    features_idxs, features_values, labels = None, None, None
    train_file = train_data_path + '/train.txt'
    print("train.txt is at: ", train_file)
    train_item_count = count_in_file_items(train_file)
    print("train_item_count is: ", train_item_count)

    features_idxs, features_values, labels = get_idx_value_label(train_file, feat_dict_)

    # train the model per batcb
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        # get the start and end index for the current batch
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        ed_idx = min(ed_idx, train_item_count - 1)

        batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
        batch_fea_values = features_values[st_idx:ed_idx, :]
        batch_labels = labels[st_idx:ed_idx, :]

        # convert the numpy format to torch tensor format
        idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
        idx = idx.to(device)
        batch_fea_values = torch.from_numpy(batch_fea_values)
        batch_labels = torch.from_numpy(batch_labels)
        value = batch_fea_values.to(device, dtype=torch.float32)
        target = batch_labels.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # 1. FORWARD PASS
        output = model(idx, value)
        
        
        loss = F.binary_cross_entropy_with_logits(output, target)

        # use regularization if necessary
        if use_reg_l1:
            for param in model.parameters():
                loss += model.reg_l1 * torch.sum(torch.abs(param))
        if use_reg_l2:
            for param in model.parameters():
                loss += model.reg_l2 * torch.sum(torch.pow(param, 2))

        # 2. BACKWARD PASS/BACK PROPOGATION
        loss.backward()

        # Grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        
        # 3. Update
        optimizer.step()
        
        print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
            epoch, batch_idx * len(idx), train_item_count,
            100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def test_model(model, test_data_path, feat_dict_, device):
    """
    evaluate the model
    :param model: The NN based model
    :param test_data_path: where the test data locates
    :param feat_dict_: a feature dict, e.g. {feature_name: idx} 
    :return:
    """
    pred_y, true_y = [], []
    features_idxs, features_values, labels = None, None, None
    test_file = test_data_path + '/test.txt'
    features_idxs, features_values, labels = get_idx_value_label(test_file, feat_dict_, shuffle=False)
    test_loss = 0
    test_item_count = count_in_file_items(test_file)
    with torch.no_grad():
        for batch_idx in range(math.ceil(test_item_count / BATCH_SIZE)):
            st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
            ed_idx = min(ed_idx, test_item_count - 1)

            batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
            batch_fea_values = features_values[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx, :]

            idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
            idx = idx.to(device)
            batch_fea_values = torch.from_numpy(batch_fea_values)
            batch_labels = torch.from_numpy(batch_labels)
            value = batch_fea_values.to(device, dtype=torch.float32)
            target = batch_labels.to(device, dtype=torch.float32)
           
            # 1. Forward Pass (Predict does not need BP and update)
            output = model(idx, value)

            test_loss += F.binary_cross_entropy_with_logits(output, target)

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average loss: {:.5f}'.format(test_loss))


def count_in_file_items(fname):
    count = 0
    with open(fname.strip(), 'r') as fin:
        for _ in fin:
            count += 1
    return count


def get_idx_value_label(fname, feat_dict_, shuffle=True):
    """
    read the train dataset from file, and return idx and value for each training example
    :param fname: file name
    :param feat_dict_: a feature dict, e.g. {feature_name: idx}  
    :param shuffle: shuffle the training examples
    :return: idx and value for all the training examples
    """
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    # for simplicity, get these statistic values in advance
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

    # process one training example
    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []

        # MinMax Normalization for dense feature
        for idx in continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[idx])
                feat_value.append((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])

        # process the sparse feature
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)

        return feat_idx, feat_value, [int(features[0])]

    features_idxs, features_values, labels = [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            feat_idx, feat_value, label = _process_line(line)
            features_idxs.append(feat_idx)
            features_values.append(feat_value)
            labels.append(label)

    features_idxs = np.array(features_idxs)
    features_values = np.array(features_values)
    labels = np.array(labels).astype(np.int32)

    # shuffle
    if shuffle:
        idx_list = np.arange(len(features_idxs))
        np.random.shuffle(idx_list)

        features_idxs = features_idxs[idx_list, :]
        features_values = features_values[idx_list, :]
        labels = labels[idx_list, :]
    return features_idxs, features_values, labels