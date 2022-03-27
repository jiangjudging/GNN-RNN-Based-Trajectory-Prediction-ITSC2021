import os
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import torch
from torch_geometric.data import Data
from plot_helper import find_files, traj_plot_by_plt, x_magnitude, create_new_dir


class single_tp_data_pre():

    def __init__(self, csv_data_path='/home/xy/trajectories-0750am-0805am.csv', t_h=30, t_f=50, d_s=2):
        # print('\nworking on {} \n'.format(data_path))
        self.csv_data_path = csv_data_path
        self.df_0 = pd.read_csv(self.csv_data_path)
        self.vehs_id = [352, 353, 323, 328, 331, 332, 336, 338, 339, 340, 341, 342, 343, 346, 347]
        self.veh_id2traj_all = self.get_veh_id2traj_all()

    def load_pkl(self):
        with open('once_LC_vehs_i80.pkl', 'rb') as f:
            return pickle.load(f)

    def get_veh_id2traj_all(self):
        '''
        得到所有id的一个轨迹
        返回一个字典，key为车辆id，value为对应id的df
        '''
        print('getting trajectories of vehicles...')
        veh_IDs = set(self.df_0['Vehicle_ID'].values)
        veh_id2traj = {}
        for vi in veh_IDs:
            vi_lane_ids = set(self.df_0[(self.df_0['Vehicle_ID'] == vi)]['Lane_ID'].values)
            vi_traj = self.df_0[(self.df_0['Vehicle_ID'] == vi)][[
                'Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc', 'Preceeding', 'Following', 'Space_Hdwy',
                'Time_Hdwy'
            ]]
            veh_id2traj[vi] = vi_traj
        print('totally {} vehicles stay in lane 1 to 8'.format(len(veh_id2traj)))
        return veh_id2traj

    def get_frames(self, df0, frm_stpt=12, frm_enpt=610):
        '''
        选出在要求区间的frame,左闭右开原则
        '''
        vehposs = df0[(df0['Frame_ID'] >= frm_stpt) & (df0['Frame_ID'] < frm_enpt)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc']]
        return vehposs

    def find_lc_frame(self, traj_i):
        traji_laneid = traj_i['Lane_ID'].values
        traji_frameid = traj_i['Frame_ID'].values

        lc_idx = np.nonzero(traji_laneid[1:] - traji_laneid[:-1])
        lc_idx = lc_idx[0]
        if len(lc_idx) != 1:
            return 0, 0, 0
        else:
            lc_idx = lc_idx[0]
            lc_frame = traji_frameid[lc_idx]
            o_lane = traji_laneid[lc_idx]
            t_lane = traji_laneid[lc_idx + 1]
            return lc_frame, o_lane, t_lane

    def get_hist(self, veh_id, ego_id, frm_id, hist_len=30):
        '''
        返回veh_id的hist位置：
        1. 是从frm_id-30开始的31帧，左闭右开    ？？？ 这里为什么要31帧，也是不解
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于31帧的话，返回np.empty([0, 2])
        '''
        if veh_id not in self.veh_id2traj_all.keys():  # 如果不在之前找到的id内的话，就返回np.empty([0, 2])
            return np.empty([0, 2])
        ref_pos = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['Frame_ID'] == frm_id][['Local_X', 'Local_Y']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_all[veh_id], frm_stpt=frm_id - hist_len, frm_enpt=frm_id + 1)
        veh_track = veh_track[['Local_X', 'Local_Y']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < hist_len + 1:
            return np.empty([0, 2])
        return veh_track

    def get_fut(self, veh_id, ego_id, frm_id, fut_len=50):
        '''
        返回veh_id的fut位置：
        1. 是从frm_id+1开始的50帧，左闭右开
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于50帧的话，返回np.empty([0, 2])
        '''
        ref_pos = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['Frame_ID'] == frm_id][['Local_X', 'Local_Y']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_all[veh_id], frm_stpt=frm_id + 1, frm_enpt=frm_id + fut_len + 1)
        veh_track = veh_track[['Local_X', 'Local_Y']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < fut_len:
            return np.empty([0, 2])
        return veh_track

    def get_nbrs_one_lane(self, ego_id, frm_id, lane='left'):
        '''
        返回指定帧，指定车道的车的id：
        1. ego车道就直接从Preceeding和Following获取
        2. left和right车道的话，就先选择一个最近的车辆作为mid车辆，然后从mid车辆的Preceeding和Following获取对应车道的前后车
        '''
        if lane == 'ego':
            # 直接从id对应帧的df中获取Preceeding和Following
            possible_pre_fol_vehc = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['Frame_ID'] == frm_id][['Preceeding', 'Following']].values
            if possible_pre_fol_vehc.shape[0] == 0:
                return [0]
            tar_pre_vid, tar_fol_vid = possible_pre_fol_vehc[0]
            nbrs = [int(ego_id), int(tar_fol_vid), int(tar_pre_vid)]  # put the ego at the first place
            return nbrs

        # 对应帧和id的y坐标以及lane 的id
        ego_y, ego_lane = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['Frame_ID'] == frm_id][['Local_Y', 'Lane_ID']].values[0]

        #同一帧内左侧车道的id以及y 坐标
        if lane == 'left':
            vehs_tar_lane = self.df_0[(self.df_0['Frame_ID'] == frm_id) & (self.df_0['Lane_ID'] == int(ego_lane - 1))][['Vehicle_ID', 'Local_Y']].values

        #同一帧内右侧车道的id以及y 坐标
        if lane == 'right':
            vehs_tar_lane = self.df_0[(self.df_0['Frame_ID'] == frm_id) & (self.df_0['Lane_ID'] == int(ego_lane + 1))][['Vehicle_ID', 'Local_Y']].values
        if vehs_tar_lane.size == 0:
            return [0]

        # 把y坐标转变为与ego_y的绝对距离差，然后把距离差最小的作为中间车辆
        vehs_tar_lane[:, 1] = abs(vehs_tar_lane[:, 1] - ego_y)  # the longitudinal distance
        tar_mid_idx = np.where(vehs_tar_lane[:, 1] == np.min(vehs_tar_lane[:, 1]))[0][0]
        tar_mid_vid = vehs_tar_lane[:, 0][tar_mid_idx]
        tar_pre_vid, tar_fol_vid = self.veh_id2traj_all[tar_mid_vid][self.veh_id2traj_all[tar_mid_vid]['Frame_ID'] == frm_id][['Preceeding',
                                                                                                                               'Following']].values[0]
        nbrs = [int(tar_fol_vid), int(tar_mid_vid), int(tar_pre_vid)]
        return nbrs

    def get_nbrs_all(self, ego_id, frm_id):
        '''
        得到指定id以及frame id的周围车辆：
        1. 如果是0的话，就删除，不加入
        2. 返回的nbrs_all的第一位是target车辆，接下来就是按照论文那样的顺序
        '''
        nbrs_all = []
        for l in ['ego', 'left', 'right']:  # put the ego at the first place
            nbrs_1_lane = self.get_nbrs_one_lane(ego_id, frm_id, lane=l)
            nbrs_all += nbrs_1_lane
        # print(nbrs_all)
        # remove 0s from nbrs
        nbrs_all = [vid for vid in nbrs_all if vid != 0]
        # print('nbrs without 0s {}'.format(nbrs_all))
        return nbrs_all

    def get_data_item(self, ego_id, frm_id):
        '''
        得到指定ego_id在指定帧的数据，包括目标车辆的历史，未来轨迹，以及邻居车辆的历史轨迹
        1. hist_all，第一位就是ego车辆的历史轨迹，后面就是邻居车辆的历史轨迹（不一定是按照1-8的顺序，不存在的就没有加进来）
        2. f1就是ego车辆的未来轨迹
        3. 邻居车辆确保在30帧内都存在
        '''
        nbrs_all = self.get_nbrs_all(ego_id, frm_id)
        # print(nbrs_all)
        # initialize data
        hist_all = np.zeros([len(nbrs_all), 31, 2])
        f1 = np.zeros([1, 50, 2])

        # get the fut of ego (target)
        f = self.get_fut(ego_id, ego_id, frm_id)
        if len(f) == 0:
            return np.empty([0, 31, 2]), np.empty([0, 50, 2])
        f1[0] = f

        # get hist of all vehicles (nbrs and ego)
        for i, vi in enumerate(nbrs_all):
            h = self.get_hist(vi, ego_id, frm_id)
            if len(h) == 0:  # 只要有一个neighbor不符合要求就返回空值
                return np.empty([0, 31, 2]), np.empty([0, 50, 2])
            else:
                hist_all[i] = h

        # edges of a star-like graph

        return hist_all, f1

    def turn_data_item_to_pyg(self, hist_all, fut_gt):
        '''
        建图：
        1. 以所有车辆作为节点，他们的历史轨迹作为节点特征
        2. 建立星型的边，ego在中心，与邻居车辆有个双向的边，然后与ego自身也有个边(self loop)
        '''
        # node features and the ground truth future trajectory
        Node_f = torch.from_numpy(hist_all)  # node features
        fut_gt = torch.from_numpy(fut_gt)  # the label, ground truth of future traj

        # edges, star-like with self loop
        edge_st_pt = torch.tensor([i for i in range(hist_all.shape[0])]).unsqueeze(0).long()
        edge_nd_pt = torch.zeros(hist_all.shape[0]).unsqueeze(0).long()
        in_edges = torch.cat((edge_st_pt, edge_nd_pt), dim=1)  # tensor([[0, 1, 2, 3, 0, 0, 0, 0]])
        out_edges = torch.cat((edge_nd_pt, edge_st_pt), dim=1)  # tensor([[0, 0, 0, 0, 0, 1, 2, 3]])
        # tensor([[0, 1, 2, 3, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 2, 3]])
        Edges = torch.cat((in_edges, out_edges), dim=0)

        # turn to pyg data
        one_pyg_data = Data(node_feature=Node_f, edge_index=Edges, y=fut_gt)

        return one_pyg_data

    def preprocess_data(self):

        data_k = self.csv_data_path.split('trajectories-')[1].split('.')[0]

        # vehs_LC_once = self.once_lc_vehs  #这是once_lc_veh_selector.py保存的once_LC_vehs_us101.pkl

        print('preprocessing data of {}, {} once LC vehicles in total'.format(data_k, len(self.vehs_id)))
        for ego_id in self.vehs_id:
            # break
            print(f"{data_k},ego_id: {ego_id}")
            traji = self.veh_id2traj_all[ego_id]  # ego_id的轨迹
            # lc_f, ol, tl = self.find_lc_frame(traji)  # 换道的帧id

            # min_frm_id = min(set(traji['Frame_ID'].values))
            # max_frm_id = max(set(traji['Frame_ID'].values))
            frms_id = traji['Frame_ID'].values
            if (len(frms_id) > 81):
                out_dir = create_new_dir("/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/ngsim_single_datasets",
                                         f"stp{data_k}_v{ego_id}")
            # 只要变道前后130帧的结果？？？ 这是为什么呢-->是因为要balanced data吗
            for frm_id in frms_id[31:-50]:
                Hist, futGT = self.get_data_item(ego_id, frm_id)
                pyg_data_item = self.turn_data_item_to_pyg(Hist, futGT)

                if pyg_data_item.node_feature.shape[0] == 0 or pyg_data_item.y.shape[0] == 0:
                    continue
                data_name = f"{out_dir}/stp{data_k}_v{ego_id}_f{frm_id}.pyg"
                # data_name = f'stp_data_i80/stp{}_v{}_f{}.pyg'.format(data_k, ego_id, frm_id)
                torch.save(pyg_data_item, data_name)


if __name__ == '__main__':

    csv_data_path_list = ['/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/data_us101/trajectories-0805am-0820am.csv']
    for data_path in csv_data_path_list[0:]:
        single_data_pre = single_tp_data_pre(csv_data_path=data_path)
        single_data_pre.preprocess_data()
        # single_data_pre.get_nbrs_all(ego_id=33, frm_id=300)
        # break