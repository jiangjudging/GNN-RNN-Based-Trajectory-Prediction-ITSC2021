import imp
import os
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
import pickle, json
import time
import torch
from torch_geometric.data import Data
from plot_helper import find_files
from tqdm import tqdm

ibeo_csv_dir = "ld_data/processed_ibeo_csvs"
ibeo_csv_dict = {
    1: 'OD_LiangDao_20220318_9988_144017_fusion_00_cut.csv',
    2: 'OD_LiangDao_20220318_9988_151727_fusion_00_cut.csv',
    3: 'OD_LiangDao_20220318_9988_164021_fusion_00_cut.csv',
    4: "OD_LiangDao_20220318_9988_164021_fusion_01_cut.csv",
    5: "OD_LiangDao_20220318_9988_171225_fusion_00_cut.csv"
}


class single_tp_data_pre():

    def __init__(self, csv_data_path, ds_id_dict, save_path, t_h=30, t_f=50, d_s=2, sample_frm_int=1, x_min=-40, x_max=60, y_min=-9, y_max=9):
        self.csv_data_path = csv_data_path
        # self.csv_name = csv_data_path.split('/')[-1][:64]
        self.csv_name = csv_data_path.split('/')[-1][:-3]
        self.save_path = save_path
        ds_id = -1
        for k, v in ds_id_dict.items():
            if v[:-3] == self.csv_name:
                ds_id = int(k)
                break

        if ds_id == -1:
            raise ValueError(f"Wrong dataset id")

        print(f"Dataset id: {ds_id}")

        self.df_0 = pd.read_csv(self.csv_data_path)

        # 给id加一个大的值，以便用来区分
        # self.df_0.obj_id += base_id

        # 得到每个id的运动轨迹，这里没有进行过滤
        self.veh_id2traj_all = self.get_veh_id2traj_all()

        # 初始化
        self.ds_id = ds_id
        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.sample_frm_int = sample_frm_int
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_veh_id2traj_all(self):
        '''
        得到所有id的一个轨迹
        返回一个字典，key为车辆id，value为对应id的df
        '''
        print('getting trajectories of vehicles...')
        veh_IDs = set(self.df_0['obj_id'].values)
        veh_id2traj = {}
        for vi in veh_IDs:
            vi_traj = self.df_0[(self.df_0['obj_id'] == vi)][['frame_id', 'obj_id', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
            # if (len(vi_traj) < 81):
            #     continue
            veh_id2traj[vi] = vi_traj
        print('totally {} vehicles'.format(len(veh_id2traj)))
        return veh_id2traj

    def get_frames(self, df0, frm_stpt=12, frm_enpt=610):
        '''
        选出在要求区间的frame,左闭右开原则
        '''
        vehposs = df0[(df0['frame_id'] >= frm_stpt) &
                      (df0['frame_id'] < frm_enpt)][['frame_id', 'obj_id', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
        return vehposs

    def get_ego_hist(self, ego_id, frm_id, hist_len=30):
        '''
        返回ego车辆在frm_id的坐标，作为ref_pos，orie_w，以及历史hist_len帧的位置（相对于ref_pos）
        '''
        if ego_id not in self.veh_id2traj_all.keys():  # 如果不在之前找到的id内的话，就返回np.empty([0, 2])
            return None, None, np.empty([0, 2])
        # 先计算ref_pos以及 orie_w
        ego_frame_df = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['frame_id'] == frm_id]
        ref_pos = ego_frame_df[['GlobalX', 'GlobalY']].values[0]
        ego_loc_pos = ego_frame_df[['LocalX', 'LocalY']].values[0]

        # 得到历史轨迹
        veh_track = self.get_frames(self.veh_id2traj_all[ego_id], frm_stpt=frm_id - hist_len, frm_enpt=frm_id + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values

        # 减去ref_pos，得到相对于ref_pos的位置
        veh_track = veh_track - ref_pos

        # 如果得到的帧数小于hist_len + 1，也返回empty
        if len(veh_track) < hist_len + 1:
            return None, None, np.empty([0, 2])
        return ref_pos, ego_loc_pos, veh_track

    def get_nbr_hist(self, veh_id, frm_id, ref_pos, hist_len=30):
        '''
        返回veh_id的hist位置：
        1. 是从frm_id-30开始的31帧，左闭右开    ？？？ 这里为什么要31帧，也是不解
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于31帧的话，返回np.empty([0, 2])
        '''
        if veh_id not in self.veh_id2traj_all.keys():  # 如果不在之前找到的id内的话，就返回np.empty([0, 2])
            return np.empty([0, 2])

        # ref_pos = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['frame_id'] == frm_id][['GlobalX', 'GlobalY']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_all[veh_id], frm_stpt=frm_id - hist_len, frm_enpt=frm_id + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < hist_len + 1:
            return np.empty([0, 2])
        return veh_track

    def get_fut(self, veh_id, frm_id, ref_pos, fut_len=50):
        '''
        返回veh_id的fut位置：
        1. 是从frm_id+1开始的50帧，左闭右开
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于50帧的话，返回np.empty([0, 2])
        '''
        veh_track = self.get_frames(self.veh_id2traj_all[veh_id], frm_stpt=frm_id + 1, frm_enpt=frm_id + fut_len + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < fut_len:
            return np.empty([0, 2])
        return veh_track

    def is_in_roi(self, ego_loc_pos, obj_loc_pos):
        return (self.x_min <= obj_loc_pos[0] - ego_loc_pos[0] <= self.x_max) and (self.y_min <= obj_loc_pos[1] - ego_loc_pos[1] <= self.y_max)

    def get_nbr_ids(self, ego_id, frm_id, ego_loc_pos):
        '''
        得到指定id以及frame id的周围车辆id：
        周围车辆指的是在frm_id时，ego车辆附近的一个矩形区域内的车辆
        矩形区域由四条边（x_min, x_max, y_min, y_max）确定
        '''
        nbr_ids = []
        # 得到这一帧的所有obj id
        all_nbr_ids = self.df_0[self.df_0.frame_id == frm_id].obj_id.unique()

        # 遍历所有id
        for id in all_nbr_ids:
            # 如果是ego_id，就跳过
            if id == ego_id:
                continue
            # 得到id在此帧的位置
            obj_loc_pos = self.veh_id2traj_all[id][self.veh_id2traj_all[id]['frame_id'] == frm_id][['LocalX', 'LocalY']].values[0]
            if (self.is_in_roi(ego_loc_pos, obj_loc_pos)):
                nbr_ids.append(id)
        return nbr_ids

    def get_nbrs_hist(self, ego_id, frm_id, ref_pos, ego_loc_pos):
        '''
        得到指定id以及frame id的周围车辆轨迹：
        周围车辆指的是在frm_id时，ego车辆附近的一个矩形区域内的车辆
        矩形区域由四条边（x_min, x_max, y_min, y_max）确定
        
        流程：
        1. 先得到frm_id是的周围车辆id
        2. 遍历周围车辆，得到hist_track
        '''
        nbr_ids = self.get_nbr_ids(ego_id, frm_id, ego_loc_pos)
        nbrs_hist = []
        nbr_final_ids = []
        for nbr_id in nbr_ids:
            nbr_hist = self.get_nbr_hist(nbr_id, frm_id, ref_pos)
            if len(nbr_hist) > 0:
                nbrs_hist.append(nbr_hist)
                nbr_final_ids.append(nbr_id)

        # print('nbrs without 0s {}'.format(nbrs_all))
        return nbrs_hist, nbr_final_ids

    def get_data_item(self, ego_id, frm_id):
        '''
        得到指定ego_id在指定帧的数据，包括目标车辆的历史，未来轨迹，以及邻居车辆的历史轨迹
        1. hist_all，第一位就是ego车辆的历史轨迹，后面就是邻居车辆的历史轨迹
        2. f1就是ego车辆的未来轨迹
        3. 邻居车辆确保在30帧内都存在
        '''
        # 得到ego车辆的ref_pos,ego_loc_pos,以及他的历史轨迹ego_hist
        ref_pos, ego_loc_pos, ego_hist = self.get_ego_hist(ego_id, frm_id)
        if ref_pos is None:
            return np.empty([0, 31, 2]), np.empty([0, 50, 2]), []

        # 得到周围车辆的历史轨迹
        nbrs_hist, nbr_ids = self.get_nbrs_hist(ego_id, frm_id, ref_pos, ego_loc_pos)
        hist_ids = [ego_id] + nbr_ids
        # print(nbrs_all)

        # initialize data
        hist_all = np.zeros([len(nbrs_hist) + 1, 31, 2])
        hist_all[0] = ego_hist  # 第一个必须是ego历史轨迹
        f1 = np.zeros([1, 50, 2])

        # get the fut of ego (target)
        f = self.get_fut(ego_id, frm_id, ref_pos)
        if len(f) == 0:
            return np.empty([0, 31, 2]), np.empty([0, 50, 2]), []
        # print(ego_id, frm_id)
        f1[0] = f

        # get hist of all vehicles (nbrs and ego)
        for i, nbr_hist in enumerate(nbrs_hist):
            hist_all[i + 1] = nbr_hist

        # edges of a star-like graph

        return hist_all, f1, hist_ids

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
        total_cnt = 0
        samples = []
        for ego_id, ego_traj in self.veh_id2traj_all.items():

            frm_ids = ego_traj['frame_id'].to_numpy()
            min_frm_id = np.min(frm_ids)
            max_frm_id = np.max(frm_ids)

            #如果存活时间小于81帧的，就直接跳过
            if (max_frm_id - min_frm_id < 81):
                continue

            print(f"ego_id: {ego_id}: cnt: {len(frm_ids)-81}")

            # 把[31:-50]帧的每一帧都作为目标帧
            for frm_id in tqdm(frm_ids[31:-50:self.sample_frm_int]):
                Hist, futGT, hist_ids = self.get_data_item(ego_id, frm_id)
                pyg_data_item = self.turn_data_item_to_pyg(Hist, futGT)

                if pyg_data_item.node_feature.shape[0] == 0 or pyg_data_item.y.shape[0] == 0:
                    continue
                # print(f"ego_id: {ego_id}    frm_id: {frm_id}")
                sample = [self.ds_id, frm_id] + hist_ids
                samples.append(sample)
                # data_name = f"ld_data_pygs/v{ego_id}_f{frm_id}.pyg"
                # torch.save(pyg_data_item, data_name)
                total_cnt += 1

        with open(f"{self.save_path}/{self.csv_name}", "wb") as fp:  #Pickling
            pickle.dump(samples, fp)

        print(f"total cnt: {total_cnt}")


if __name__ == '__main__':
    # ibeo 预处理
    sample_save_path = "/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/ld_data/processed_ibeo_samples_list"
    for key, value in ibeo_csv_dict.items():
        csv_data_path = os.path.join(ibeo_csv_dir, value)
        ibeo_pre = single_tp_data_pre(csv_data_path, ibeo_csv_dict, sample_save_path)
        ibeo_pre.preprocess_data()
    # import json
    # dataset_dir = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis"
    # ego_csvs = find_files(dataset_dir, recursive=True, suffix="ego.csv")
    # #LIDAR_LJ02766_20210915_122721_G260-PDX-006-001-052_000000-000047_LD_final_OD_MERGE_OPP_ego
    # base_id_dict = {}
    # base = 1
    # for ego_csv in ego_csvs:
    #     ego_csv_name = ego_csv.split('/')[-1][:64]
    #     print(ego_csv_name)
    #     base_id_dict[base] = ego_csv_name
    #     base += 1
    # print(base_id_dict)
    # with open("dataset_id_mapping.json", "w") as outfile:
    #     json.dump(base_id_dict, outfile, indent=4)

    # with open("dataset_id_mapping.json", "r") as outfile:
    #     a = json.load(outfile, parse_int=True)
    #     print(a)
    # merge_10hz_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP_fusion_10hz.csv"

    # dataset_dirpath = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis"
    # import shutil
    # dataset_dirpath = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis"
    # to_cp_dirpath = "/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/ld_data/processed_csvs"
    # fusion_10hz_csvs = find_files(dataset_dirpath, recursive=True, suffix='fusion_10hz.csv')
    # ds_id_map = "dataset_id_mapping.json"
    # ds_id_dict = {}

    # with open(ds_id_map, "r") as outfile:
    #     ds_id_dict = json.load(outfile)
    # for fusion_10hz_csv in fusion_10hz_csvs:
    #     # tp_data_pre = single_tp_data_pre(fusion_10hz_csv, ds_id_dict)
    #     # print(tp_data_pre.csv_name)
    #     # tp_data_pre.preprocess_data()
    #     shutil.copy(fusion_10hz_csv, to_cp_dirpath)

    # with open("ld_data/processed_samples_list/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060", "rb") as fp:  # Unpickling
    #     b = pickle.load(fp)
    # print(len(b))
    # print(b[0:200])
    # tp_data_pre.preprocess_data()

    # ego_id = 10604
    # frm_id = 5180

    # ref_pos, ego_loc_pos, ego_hist = tp_data_pre.get_ego_hist(ego_id, frm_id)
    # hist_all, f, hist_id = tp_data_pre.get_data_item(ego_id, frm_id)

    # print(hist_all.shape, hist_all)
    # print(f.shape, f)
    # print(hist_id)

    # hist_all_glb = hist_all + ref_pos
    # f_glb = f + ref_pos
    # print(hist_all_glb.shape, hist_all_glb)
    # print(f_glb.shape, f_glb)

    # csv_data_path_list = ['data_i80/trajectories-0400pm-0415pm.csv', 'data_i80/trajectories-0500pm-0515pm.csv', 'data_i80/trajectories-0515pm-0530pm.csv']
    # for data_path in csv_data_path_list[0:]:
    #     single_data_pre = single_tp_data_pre(csv_data_path=data_path)
    #     single_data_pre.preprocess_data()
    # single_data_pre.get_nbrs_all(ego_id=33, frm_id=300)
    # break
