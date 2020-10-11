import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.train_model_util import train_test_model

EPOCHS = 10
BATCH_SIZE = 2048
# TODO: to deprecate abs path
DATA_DIR = '../data/CriteoDataset/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PyTorch implementation of DCN
Reference:
Deep & Cross Network for Ad Click Predictions
Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang
"""

class DCN(nn.Module):
    def __init__(self, num_feat, num_field, dropout_deep, dropout_cross,
                 reg_l1=0.01, reg_l2=0.01, l_c = 3, layer_sizes=[400, 400, 400], embedding_size=10):
        super().__init__()  
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2                  
        self.num_feat = num_feat                                                      # denote as M
        self.num_field = num_field                                                    # denote as F
        self.embedding_size = embedding_size                                          # denote as K
        self.layer_sizes = layer_sizes
        self.l_c = l_c
        self.dropout_deep = dropout_deep
        self.dropout_cross = dropout_cross

        assert len(self.dropout_cross) == self.l_c, "Please specify dropout for each layer of cross net."

        # cross parameters
        cross_dim = (self.num_field-13) * self.embedding_size + 13
        for i in range(0, self.l_c):
            setattr(self, 'linear_cross_' + str(i), nn.Linear(cross_dim, 1))
            setattr(self, 'dropout_cross_' + str(i), nn.Dropout(dropout_cross[i]))

        # dense parameters embedding
        self.dense_weights = nn.Embedding(13 + 1, 1)                                  # 14 * 1
        nn.init.xavier_uniform_(self.dense_weights.weight)

        # sparse parameters embedding
        self.feat_embeddings = nn.Embedding(num_feat, embedding_size)                 # M * K
        nn.init.xavier_uniform_(self.feat_embeddings.weight)

        # deep parameters
        all_dims = [cross_dim] + layer_sizes
        for i in range(1, len(layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))

        # FC as last layer
        self.fc = nn.Linear(cross_dim + all_dims[-1], 1)
        

    def forward(self, feat_index, feat_value):
        # feat_index: None * F
        # feat_value: None * F
        feat_value = torch.unsqueeze(feat_value, dim=2)                                       # None * F * 1
        dense_index = feat_index[:, :13]
        sparse_index = feat_index[:, 13:]
        dense_value = feat_value[:, :13, :]
        sparse_value = feat_value[:, 13:, :]

        dense_weights = self.dense_weights(dense_index)                                       # None * 13 * 1
        dense_weight_value = torch.mul(dense_weights, dense_value)                            # None * 13 * 1
        x_dense_data = torch.sum(dense_weight_value, dim=2)                                   # None * 13
        sparse_weight = self.feat_embeddings(sparse_index)                                    # None * F * K
        x_sparse_data = sparse_weight * sparse_value                                          # None * F * K (broadcast)
        x_sparse_data = x_sparse_data.reshape(-1, (self.num_field-13) * self.embedding_size)  # None * (F * K)

        concat_input = torch.cat((x_dense_data, x_sparse_data), dim=1)

        # Cross Network
        x_0 = concat_input
        res = x_0
        for i in range(self.l_c):
            res = getattr(self, 'linear_cross_' + str(i))(res) * x_0 + res
            res = getattr(self, 'dropout_cross_' + str(i))(res)          

        # Deep Network
        y_deep = x_0
        for i in range(1, len(self.layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        concat_input = torch.cat((res, y_deep), dim=1)
        output = self.fc(concat_input)
        return output


if __name__ == '__main__':
    train_data_path, test_data_path = DATA_DIR + 'train_data/', DATA_DIR + 'test_data/'
    feat_dict_file = DATA_DIR + 'aid_data/feat_dict_10.pkl2'
    feat_dict_ = pickle.load(open(feat_dict_file, 'rb'))

    dcn = DCN(num_feat=len(feat_dict_) + 1, num_field=39, l_c=7,
                    dropout_deep=[0.5, 0.5, 0.5, 0.5], dropout_cross=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    layer_sizes=[400, 400, 400], embedding_size=10).to(DEVICE)

    train_test_model(dcn, DEVICE, train_data_path, test_data_path, feat_dict_)