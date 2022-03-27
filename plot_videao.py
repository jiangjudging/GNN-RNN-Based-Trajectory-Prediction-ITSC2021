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

import math
import time
from plot_helper import find_files, traj_plot_by_plt, x_magnitude, create_new_dir
import os.path as osp

import matplotlib.pyplot as plt
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage


def test_a_model(model_to_test, testDataloader):
    model_to_test.eval()
    fut_preds_dict = {}

    with torch.no_grad():
        print('Testing no grad')
        # val_running_loss = 0.0
        # cnt = 0
        # cnt2 = 0
        for i, data in enumerate(testDataloader):
            # down-sampling data
            data.x = data.x[:, ::2, :]
            data.y = data.y[:, 4::5, :]

            # predict
            fut_pred = model_to_test(data.to(args['device']))
            fut_preds_dict[data.fname[0]] = fut_pred.detach().cpu()
            # print(fut_pred.shape)
            # calculate loss
    # print(len(fut_preds))
    # print(fut_preds.shape)
    return fut_preds_dict


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
        parser.add_argument('-k', '--gpu', type=str, default='1', help="the GPU to be used")
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

    test_net.load_state_dict(torch.load("trained_models/2022_01_27_14_43_GR_GAT_GRU_h30f10_d3s_16_1.0s.tar"))
    test_net.to(device)

    to_draw_ids = [337, 352, 353, 323, 328, 329, 331, 332, 336, 338, 339, 340, 341, 342, 343, 344, 346, 347]
    for id in to_draw_ids:
        test_set = STP_GR_Dataset(data_path=f'ngsim_single_datasets/stp0805am-0820am_v{id}', scenario_names=['stp0805am-0820am'])

        # if args['single_or_multiple'] == 'single_tp':
        #     test_set = STP_GR_Dataset(data_path='stp_data_all/stp0805am-0820am_v337',
        #                               scenario_names=[
        #                                   'stp0750am-0805am',
        #                                   'stp0805am-0820am',
        #                                   'stp0820am-0835am',
        #                                   'stp0400pm-0415pm',
        #                                   'stp0500pm-0515pm',
        #                                   'stp0515pm-0530pm',
        #                               ])
        # elif args['single_or_multiple'] == 'multiple_tp':
        #     pass

        # torch.set_num_threads(1)
        testDataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        fut_preds = test_a_model(test_net, testDataloader)
        # np.save("imgs/stp0805am-0820am_v337/v337_fur_pred_dict.npy", fut_preds)
        save_obj_pkl(fut_preds, f"fur_preds/fut_preds_dict_stp0805am-0820am_v{id}")
