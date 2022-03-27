from __future__ import print_function
from datetime import date, datetime
import os
import sys
import argparse
import random
import pickle
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from utils import maskedMSE, maskedMSETest, maskedMSEFinalTest

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from stp_r_model import STP_R_Net
from stp_g_model import STP_G_Net
from stp_gr_model import STP_GR_Net

from stp_gr_dataset import STP_GR_Dataset
from ld_dataset import LD_Dataset, get_smp_list_ds_df_dict

import math
import time
from plot_helper import find_files, traj_plot_by_plt, x_magnitude, create_new_dir, ld_traj_plot_by_plt
import os.path as osp


def test_a_model(model_to_test, testDataloader):
    model_to_test.eval()
    x_lossVals = torch.zeros(10)
    y_lossVals = torch.zeros(10)
    lossVals = torch.zeros(10)
    counts = torch.zeros(10)
    with torch.no_grad():
        print('Testing no grad')
        # val_running_loss = 0.0
        cnt = 0
        cnt2 = 0
        for i, data in enumerate(testDataloader):
            # down-sampling data
            data.x = data.x[:, ::2, :]
            data.y = data.y[:, 4::5, :]

            # predict
            fut_pred = model_to_test(data.to(args['device']))
            print(data.ds_id)
            print(fut_pred.shape)
            # calculate loss
            for j in range(len(fut_pred)):
                fut_gt = data.y[j:j + 1, :, :]
                fut_pred_i = fut_pred[j:j + 1]

                fut_gt = fut_gt
                fut_pred_i = fut_pred_i
                x = data.x[data.batch == j]
                if data.ds_id[j].cpu() == 9 and data[j].tgt_id == 364:
                    ld_traj_plot_by_plt(x.cpu(), fut_gt.cpu(), fut_pred_i.cpu(),
                                        f"imgs/ld_eval_imgs/ds9_id364/ld_ds{data.ds_id[j].cpu()}_id{data[j].tgt_id}_frmid{data[j].frm_id}.png")

                    cnt += 1
                    if (cnt >= 20):
                        return

                # x_mag = max(fut_gt[0, :, 0]) - min(fut_gt[0, :, 0])
                # if (x_mag) > 3:
                #     print(cnt, data.fname[j])
                #     traj_plot_by_plt(x.cpu(), fut_gt.cpu(), fut_pred_i.cpu(), f"./imgs/id_test_rdm/lc_{data.fname[j]}.png")
                #     cnt += 1
                #     if (cnt >= 10):
                #         return
                # elif cnt2 < 5:
                #     print(cnt2, data.fname[j])
                #     traj_plot_by_plt(x.cpu(), fut_gt.cpu(), fut_pred_i.cpu(), f"./imgs/id_test_rdm/lk_{data.fname[j]}.png")
                #     cnt2 += 1

    x_rmse_loss_m = torch.pow(x_lossVals / counts, 0.5) * 0.3048
    y_rmse_loss_m = torch.pow(y_lossVals / counts, 0.5) * 0.3048
    rmse_loss_m = torch.pow(lossVals / counts, 0.5) * 0.3048

    print(f"x_rmse_loss_m: {x_rmse_loss_m}")
    print(f"y_rmse_loss_m: {y_rmse_loss_m}")
    print(f"rmse_loss_m: {rmse_loss_m}")
    return x_rmse_loss_m, y_rmse_loss_m, rmse_loss_m


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
            test_net = STP_GR_Net(args)
        elif args['single_or_multiple'] == 'multiple_tp':
            print('loading {} model'.format(args['net_type']))
            # test_net = MTP_GR_Net(args)
    elif args['net_type'] == 'R':
        print('loading {} model'.format(args['net_type']))
        test_net = STP_R_Net(args)
    elif args['net_type'] == 'G':
        print('loading {} model'.format(args['net_type']))
        test_net = STP_G_Net(args)
    else:
        print('\nselect a proper model type!\n')
    test_net.to(args['device'])

    pytorch_total_params = sum(p.numel() for p in test_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))
    print(test_net)
    pp.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))

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

    # tic = time.time()
    test_net.load_state_dict(torch.load("trained_models/ld_2022_03_17_22_39_GR_GAT_GRU_h30f10_d3s_16.tar"))
    test_net.to(device)
    test_loss_ep = test_a_model(test_net, trainDataloader)
