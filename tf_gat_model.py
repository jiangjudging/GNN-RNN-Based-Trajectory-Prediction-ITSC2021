import os.path as osp
import torch, math
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TF_GAT_Model(torch.nn.Module):
    ''' 
        Shared layers:
            self.ip_emb
            self.enc_rn
            self.dyn_emb
            self.op
            self.leaky_relu

            self.LSTM_Encoder
            self.decode
        '''

    def __init__(self, args):
        super().__init__()
        self.args = args

        # tf_encoder
        encoder_layer = TransformerEncoderLayer(d_model=16, nhead=2, dropout=0.0)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=4)
        self.pos_encoder = PositionalEncoding(16, 0)
        self.encoder_fc = nn.Linear(16, 2)

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        # Encoder LSTM
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # Decoder LSTM
        self.dec_rnn = torch.nn.LSTM(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        # Output layers:
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)

        # GAT layers
        self.gat_conv1 = GATConv(self.args['encoder_size'],
                                 self.args['encoder_size'],
                                 heads=self.args['num_gat_heads'],
                                 concat=self.args['concat_heads'],
                                 dropout=0.0)
        self.gat_conv2 = GATConv(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'],
                                 self.args['encoder_size'],
                                 heads=self.args['num_gat_heads'],
                                 concat=self.args['concat_heads'],
                                 dropout=0.0)
        # fully connected
        self.nbrs_fc = torch.nn.Linear(
            int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'],
            1 * self.args['encoder_size'])

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        """ Encode sequential features of all considered vehicles 
            Hist: history trajectory of all vehicles
        """
        # print("Hist.shape", Hist.shape)
        # _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        # # print("Hist_Enc.shape", Hist_Enc.shape)
        # Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        # print("Hist_Enc.shape", Hist_Enc.shape)
        # Hist = Hist.permute(1, 0, 2).cuda()
        # print(Hist.shape)
        Hist_Enc = self.encoder_fc(self.transformer_encoder(self.pos_encoder(self.leaky_relu(self.ip_emb(Hist)))))
        Hist_Enc = torch.flatten(Hist_Enc, start_dim=1)
        # print(Hist_Enc.shape)
        return Hist_Enc

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        # elif self.args['single_or_multiple'] == 'multiple_tp':
        #     target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0:data_pyg.num_target_v[i]]) for i in range(data_pyg.num_graphs)]
        #     target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        ########################################################################

        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)

        # Interaction
        # 这里只传入lstm编码以后的特征以及边的连接情况
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)
        # print("fwd_tar_GAT_Enc.shape", fwd_tar_GAT_Enc.shape)

        # get the lstm features of target vehicles
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]
        # print("fwd_tar_GAT_Enc.shape", fwd_tar_GAT_Enc.shape)

        # Combine Individual and Interaction features
        enc = torch.cat((fwd_tar_LSTM_Enc, fwd_tar_GAT_Enc), 1)
        # print("enc.shape", enc.shape)
        # Decode
        fut_pred = self.decode(enc)
        return fut_pred

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        # print('hist_enc {}'.format(hist_enc))
        # print('node_matrix {}'.format(node_matrix))
        # print('edge_idx {}'.format(edge_idx))
        # GAT conv
        # print("node_matrix.shape", node_matrix.shape)
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        # print("gat_feature.shape", gat_feature.shape)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        # print("gat_feature.shape", gat_feature.shape)
        # print('gat_feature : {}'.format(gat_feature.shape))

        # get target node's GAT feature
        # print('gat_feature {}'.format(gat_feature.shape))
        # print('target_index {}'.format(target_index.shape))
        target_gat_feature = gat_feature[target_index]
        # print("target_gat_feature.shape", target_gat_feature.shape)

        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))
        # print("GAT_Enc.shape", GAT_Enc.shape)

        return GAT_Enc

    def decode(self, enc):
        # print(enc.shape)
        enc = enc.unsqueeze(1)
        # print('enc : {}'.format(enc.shape))
        # print(enc.shape)
        enc = enc.repeat(1, self.args['out_length'], 1)
        # print('enc : {}'.format(enc.shape))
        # print(enc.shape)
        # enc = enc.permute(1,0,2)
        h_dec, _ = self.dec_rnn(enc)
        # print('h_dec shape {}'.format(h_dec.shape))
        # h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        # print(fut_pred.shape)
        # fut_pred = fut_pred.permute(1, 0, 2)
        # print()
        return fut_pred
