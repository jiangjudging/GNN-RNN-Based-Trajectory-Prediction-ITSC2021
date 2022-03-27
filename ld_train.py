from __future__ import print_function
from datetime import date, datetime
import os
import sys
import argparse
import random
import pickle
import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from utils import maskedMSE, maskedMSETest
from plot_helper import find_files

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from stp_r_model import STP_R_Net
from stp_g_model import STP_G_Net
from stp_gr_model import STP_GR_Net
# from mtp_gr_model import MTP_GR_Net

# from stp_gr_dataset import STP_GR_Dataset
from ld_dataset import LD_Dataset, get_smp_list_ds_df_dict
# from mtp_gr_dataset import MTP_GR_Dataset

import math
import time
from torch.utils.tensorboard import SummaryWriter

log = SummaryWriter()


def train_a_model(model_to_tr, num_ep=1):
    model_to_tr.train()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    train_running_loss = 0.0
    avg_tr_time = 0
    for i, data in enumerate(trainDataloader):
        st_time = time.time()
        # down-sampling data
        data.x = data.x[:, ::2, :]
        data.y = data.y[:, 4::5, :]

        optimizer.zero_grad()
        # forward + backward + optimize
        fut_pred = model_to_tr(data.to(args['device']))

        op_mask = torch.ones(data.y.shape)
        train_l = maskedMSE(fut_pred, data.y, op_mask)

        train_l.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 10)
        optimizer.step()
        train_running_loss += train_l.item()

        batch_time = time.time() - st_time
        avg_tr_time += batch_time

        if i % 1000 == 999:  # print every 1000 mini-batches
            eta = avg_tr_time / 1000 * (len(trainDataloader) - i)
            print('ep {}, {} batches, {} - {} | ETA(s): {}'.format(num_ep, i + 1, 'maskedMSE', round(train_running_loss / 1000, 4), int(eta)))
            log.add_scalar("Loss/Train", train_running_loss, (num_ep - 1) * len(trainDataloader) + i)
            train_running_loss = 0.0
            avg_tr_time = 0
    scheduler.step()
    return round(train_running_loss / (i % 1000), 4)


# def val_a_model(model_to_val, num_ep):
#     model_to_val.eval()
#     lossVals = torch.zeros(10)
#     counts = torch.zeros(10)
#     with torch.no_grad():
#         print('Testing no grad')
#         # val_running_loss = 0.0
#         for i, data in enumerate(valDataloader):
#             # down-sampling data
#             data.x = data.x[:, ::2, :]
#             data.y = data.y[:, 4::5, :]

#             # predict
#             fut_pred = model_to_val(data.to(args['device']))

#             # calculate loss
#             fut_pred = fut_pred.permute(1, 0, 2)
#             ff = data.y.permute(1, 0, 2)

#             op_mask = torch.ones(ff.shape)
#             l, c = maskedMSETest(fut_pred, ff, op_mask)

#             lossVals += l.detach()
#             counts += c.detach()

#     rmse_loss_m = torch.pow(lossVals / counts, 0.5) * 0.3048
#     print(rmse_loss_m)
#     log.add_scalars("Loss/Val", {"1s": rmse_loss_m[1], "2s": rmse_loss_m[3], "3s": rmse_loss_m[5], "4s": rmse_loss_m[7], "5s": rmse_loss_m[9]}, num_ep)
#     return rmse_loss_m


def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # command line arguments
    pp = pprint.PrettyPrinter(indent=1)

    def parse_args(cmd_args):
        """ Parse arguments from command line input
        """
        parser = argparse.ArgumentParser(description='Training parameters')
        parser.add_argument('-g', '--gnn', type=str, default='GAT', help="the GNN to be used")
        parser.add_argument('-r', '--rnn', type=str, default='GRU', help="the RNN to be used")
        parser.add_argument('-m', '--modeltype', type=str, default='GR', help="the model type [R, G, GR]")
        parser.add_argument('-b', '--histlength', type=int, default=30, help="length of history 10, 30, 50")
        parser.add_argument('-f', '--futlength', type=int, default=10, help="length of future 50")
        parser.add_argument('-k', '--gpu', type=str, default='0', help="the GPU to be used")
        parser.add_argument('-i', '--number', type=int, default=0, help="run times of the py script")

        parser.set_defaults(render=False)
        return parser.parse_args(cmd_args)

    # Parse arguments
    cmd_args = sys.argv[1:]
    cmd_args = parse_args(cmd_args)

    ## Network Arguments
    args = {}
    args['run_i'] = cmd_args.number
    args['random_seed'] = 1
    args['input_embedding_size'] = 16  # if args['single_or_multiple'] == 'single_tp' else 32
    args['encoder_size'] = 32  # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['decoder_size'] = 64  # if args['single_or_multiple'] == 'single_tp' else 128 # 128 256
    args['dyn_embedding_size'] = 32  # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['train_epoches'] = 70
    args['num_gat_heads'] = 3
    args['concat_heads'] = True  # False # True

    args['in_length'] = cmd_args.histlength
    args['out_length'] = cmd_args.futlength

    args['single_or_multiple'] = 'single_tp'  # or multiple_tp single_tp
    # args['date'] = date.today().strftime("%b-%d-%Y")
    args['date'] = f"{datetime.now():%Y_%m_%d_%H_%M}"
    args['batch_size'] = 16 if args['single_or_multiple'] == 'single_tp' else 128
    args['net_type'] = cmd_args.modeltype
    args['enc_rnn_type'] = cmd_args.rnn  # LSTM GRU
    args['gnn_type'] = cmd_args.gnn  # GCN GAT

    device = torch.device('cuda:{}'.format(cmd_args.gpu) if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # set random seeds
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    if device != 'cpu':
        print('running on {}'.format(device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        print('seed setted! {}'.format(args['random_seed']))

    # Initialize network
    if args['net_type'] == 'GR':
        if args['single_or_multiple'] == 'single_tp':
            print('loading {} model'.format(args['net_type']))
            train_net = STP_GR_Net(args)
        elif args['single_or_multiple'] == 'multiple_tp':
            print('loading {} model'.format(args['net_type']))
            # train_net = MTP_GR_Net(args)
    elif args['net_type'] == 'R':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_R_Net(args)
    elif args['net_type'] == 'G':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_G_Net(args)
    else:
        print('\nselect a proper model type!\n')
    train_net.to(args['device'])

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))
    print(train_net)
    pp.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))

    # for name, param in train_net.named_parameters():
    #     if param.requires_grad:
    #         print(name,':',param.size())

    ## Initialize optimizer and the the
    optimizer = torch.optim.Adam(train_net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[1], gamma=1.0)
    if args['single_or_multiple'] == 'multiple_tp':
        optimizer = torch.optim.Adam(train_net.parameters(), lr=0.004)  # lr 0.0035, batch_size=4 or 8.
        scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3, 6, 20, 30], gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6], gamma=1.0)

    sample_dir = "ld_data/processed_samples_list"
    csv_dir = "ld_data/processed_csvs"
    samples_list, ds_df_dict = get_smp_list_ds_df_dict(sample_dir, csv_dir)
    train_set = LD_Dataset(samples_list, ds_df_dict)

    # if args['single_or_multiple'] == 'single_tp':
    #     train_set = LD_Dataset()
    # elif args['single_or_multiple'] == 'multiple_tp':
    #     pass
    # train_set = MTP_GR_Dataset(data_path='/home/xy/gat_mtp_data_0805am_Train/') # HIST_1w FUT_1w HIST FUT
    # val_set = MTP_GR_Dataset(data_path='/home/xy/gat_mtp_data_0805am_Test/')

    torch.set_num_threads(4)
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    # valDataloader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = [1000.0] * 10
    best_val_epoch = [0 for _ in range(10)]
    for ep in range(1, args['train_epoches'] + 1):
        train_loss_ep = train_a_model(train_net, num_ep=ep)
        # val_loss_ep = val_a_model(train_net, ep)

        # Val_LOSS.append(val_loss_ep)
        Train_LOSS.append(train_loss_ep)
        save_model_to_PATH = './trained_models/ld_{}_{}_{}_{}_h{}f{}_d{}_{}.tar'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'],
                                                                                        args['in_length'], args['out_length'], '3s', args['batch_size'])
        ## save model,保存第1，2，3，4，5s最佳的模型
        # for check_time in (1, 3, 5, 7, 9):
        #     if val_loss_ep[check_time] < min_val_loss[check_time]:
        #         save_model_to_PATH = './trained_models/ld_{}_{}_{}_{}_h{}f{}_d{}_{}_{}s.tar'.format(args['date'], args['net_type'], args['gnn_type'],
        #                                                                                             args['enc_rnn_type'], args['in_length'], args['out_length'],
        #                                                                                             '3s', args['batch_size'], (check_time + 1) / 2)
        #         torch.save(train_net.state_dict(), save_model_to_PATH)
        #         min_val_loss[check_time] = val_loss_ep[check_time]
        #         best_val_epoch[check_time] = ep

        # with open(
        #         './trained_log/ld_{}-{}-{}-{}-h{}f{}-TRAINloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'],
        #                                                                           args['in_length'], args['out_length'], '3s', args['batch_size']),
        #         "a+") as file:
        #     file.write(f"epoch {ep:02d}:{str(train_loss_ep)}\n")
        # with open(
        #         './trained_log/ld_{}-{}-{}-{}-h{}f{}-VALloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'],
        #                                                                         args['in_length'], args['out_length'], '3s', args['batch_size']), "a+") as file:
        #     file.write(f"epoch {ep:02d}:{str(val_loss_ep)}\n")
        #     if ep == args['train_epoches']:
        #         for check_time in (1, 3, 5, 7, 9):
        #             file.write(f"best val result for {(check_time + 1) / 2}s is {min_val_loss[check_time]} in epoch {best_val_epoch[check_time]}\n")
        # save_obj_pkl(args, save_model_to_PATH.split('.tar')[0])

        torch.save(train_net.state_dict(), save_model_to_PATH)