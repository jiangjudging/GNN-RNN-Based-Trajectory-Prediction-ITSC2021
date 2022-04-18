import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from stp_g_model import STP_G_Net
import numpy as np


class STP_GR_seq2seq(STP_G_Net):

    def __init__(self, args):
        super(STP_GR_seq2seq, self).__init__(args)
        self.args = args
        # # Input embedding layer
        # self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        # # Encoder LSTM
        # # self.enc_rnn = torch.nn.LSTM(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # # # Vehicle dynamics embedding
        # self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # # GAT layers
        # self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # # fully connected
        # self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        # # Decoder LSTM
        # self.dec_rnn = torch.nn.GRU(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)/
        self.dec_rnn = torch.nn.GRU(2, self.args['decoder_size'], 2, batch_first=True)
        # # Output layers:
        # self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        # # Activations:
        # self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, data_pyg, teacher_forcing_ratio):

        # get target vehicles' index first
        ########################################################################
        # for single TP
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        # elif self.args['single_or_multiple'] == 'multiple_tp':
        #     target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0:data_pyg.num_target_v[i]]) for i in range(data_pyg.num_graphs)]
        #     target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n single TP or multiple TP? \n\n')
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
        fut_pred = self.decode(enc, data_pyg.y, teacher_forcing_ratio)
        return fut_pred

    def decode(self, hidden, teacher_location, teacher_forcing_ratio=0):
        hidden = hidden.unsqueeze(0)
        hidden = hidden.repeat(2, 1, 1)

        batch_size = teacher_location.shape[0]
        out_dim = 2
        self.pred_length = 10

        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        outputs = outputs.cuda()

        decoder_input = torch.zeros(batch_size, 1, 2)
        decoder_input = decoder_input.cuda()
        for t in range(self.pred_length):
            # encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
            now_out, hidden = self.dec_rnn(decoder_input, hidden)
            now_out = self.op(now_out)
            # print(decoder_input.shape)
            # print(now_out.shape, hidden.shape)
            now_out += decoder_input
            outputs[:, t:t + 1] = now_out
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = (teacher_location[:, t:t + 1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
            # decoder_input = now_out
        return outputs
