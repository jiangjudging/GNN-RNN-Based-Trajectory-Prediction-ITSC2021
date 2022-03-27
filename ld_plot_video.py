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
from ld_dataset import LD_Dataset, get_smp_list_ds_df_dict, get_tgt_smp

import math
import time
from plot_helper import find_files, traj_plot_by_plt, x_magnitude, create_new_dir, ld_traj_plot_by_plt
import os.path as osp
from collections import defaultdict


def predict(model_to_test, testDataloader, fut_dict):
    model_to_test.eval()
    with torch.no_grad():
        print('Testing no grad')
        for i, data in enumerate(testDataloader):
            # down-sampling data
            data.x = data.x[:, ::2, :]
            data.y = data.y[:, 4::5, :]

            # predict
            fut_pred = model_to_test(data.to(args['device']))

            # calculate loss
            for j in range(len(fut_pred)):
                fut_gt = data.y[j:j + 1, :, :].cpu()
                fut_pred_i = fut_pred[j:j + 1].cpu().detach().numpy()

                x = data.x[data.batch == j].cpu().detach().numpy()

                ds_id = int(data.ds_id[j].cpu())
                tgt_id = data[j].tgt_id
                frm_id = data[j].frm_id
                if ds_id not in fut_dict:
                    fut_dict[ds_id] = {}
                if tgt_id not in fut_dict[ds_id]:
                    fut_dict[ds_id][tgt_id] = {}
                fut_dict[ds_id][tgt_id][frm_id] = fut_pred_i

    return fut_dict


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
    # train_set = LD_Dataset(samples_list, ds_df_dict)

    torch.set_num_threads(4)
    test_net.load_state_dict(torch.load("trained_models/ld_2022_03_17_22_39_GR_GAT_GRU_h30f10_d3s_16.tar"))
    test_net.to(device)

    ds_id_tgt_id = [[7, 36], [5, 37]]

    fut_dict = {}
    for ds_id, tgt_id in ds_id_tgt_id:
        tgt_smp_list = get_tgt_smp(samples_list, ds_id, tgt_id)
        print(f"dataset id: {ds_id}  target_id: {tgt_id}, sample number: {len(tgt_smp_list)}")
        test_set = LD_Dataset(tgt_smp_list, ds_df_dict)
        testDataloader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        fut_dict = predict(test_net, testDataloader, fut_dict)
    print(fut_dict)
    save_obj_pkl(fut_dict, "imgs/ld_eval_imgs/fut_pred_dict/fut_dict1")
