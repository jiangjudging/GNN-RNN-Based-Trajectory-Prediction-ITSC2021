import os
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import time
import torch
from torch_geometric.data import Data

ds_csv_dict = {
    1: "data_us101/trajectories-0750am-0805am.csv",
    2: "data_us101/trajectories-0805am-0820am.csv",
    3: "data_us101/trajectories-0820am-0835am.csv",
    4: "data_i80/trajectories-0400pm-0415pm.csv",
    5: "data_i80/trajectories-0500pm-0515pm.csv",
    6: "data_i80/trajectories-0515pm-0530pm.csv"
}


class NgsimSamplesPre():

    def __init__(self, t_h=30, t_f=50, d_s=2):
        self.ds_df_dict = {k: pd.read_csv(v) for k, v in ds_csv_dict.items()}

        self.veh_id2traj_all = self.get_veh_id2traj_all()

        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s

    def load_pkl(self):
        with open('once_LC_vehs_i80.pkl', 'rb') as f:
            return pickle.load(f)

    def get_veh_id2traj_all(self):
        '''
        得到所有id的一个轨迹
        返回一个字典，key为数据集id，value为另外一个dict：key为车辆id，value为对应id的df
        '''
        print('getting trajectories of vehicles...')
        print(f"there are {len(ds_csv_dict)} datasets")

        veh_id2traj = {}
        for ds_id, ds_df in self.ds_df_dict.items():
            print(f"getting trajectories of dataset {ds_id}")
            veh_id2traj[ds_id] = {}
            veh_IDs = set(ds_df['Vehicle_ID'].values)
            print(f"there are {len(veh_IDs)} in dataset {ds_id}")
            for vi in tqdm(veh_IDs):
                vi_traj = ds_df[(ds_df['Vehicle_ID'] == vi)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID', 'Preceeding', 'Following']]
                veh_id2traj[ds_id][vi] = vi_traj

        return veh_id2traj

    def get_frames(self, df0, frm_stpt=12, frm_enpt=610):
        '''
        选出在要求区间的frame,闭区间原则
        '''
        vehposs = df0[(df0['Frame_ID'] >= frm_stpt) & (df0['Frame_ID'] <= frm_enpt)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y']]
        return vehposs

    def get_traj(self, df, ref_frm_id):
        '''
        得到指定frm区间的一个轨迹
        '''
        frm_stpt = ref_frm_id - self.t_h + 1
        frm_enpt = ref_frm_id + self.t_f
        veh_traj = self.get_frames(df, frm_stpt, frm_enpt)[['Local_X', 'Local_Y']].values
        return veh_traj

    def is_lc(self, traj, lc_threshold=1.8 / 0.3048):
        '''
        判断是否有变道
        '''
        # 横向位移
        # print(traj)
        lat_traj = traj[:, 0]
        # print(lat_traj)
        if max(lat_traj) - min(lat_traj) > lc_threshold:
            return 1
        return 0

    def get_fut(self, ds_id, veh_id, ego_id, frm_id, fut_len=50):
        '''
        返回veh_id的fut位置：
        1. 是从frm_id+1开始的50帧，左闭右开
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于50帧的话，返回np.empty([0, 2])
        '''
        ref_pos = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['Frame_ID'] == frm_id][['Local_X', 'Local_Y']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_all[ds_id][veh_id], frm_stpt=frm_id + 1, frm_enpt=frm_id + fut_len + 1)
        veh_track = veh_track[['Local_X', 'Local_Y']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < fut_len:
            return np.empty([0, 2])
        return veh_track

    def filter_nbrs(self, ds_traj, nbrs, frm_id):
        '''
        过滤掉nbr： 
        1. nbr id不能为0
        2. nbr 在frm_id以前最少要有30帧
        '''
        nbrs = [vid for vid in nbrs if vid != 0]
        nbrs = [nbr for nbr in nbrs if len(self.get_frames(ds_traj[nbr], frm_id - self.t_h, frm_id)) == self.t_h + 1]
        return nbrs

    def get_nbrs_one_lane(self, ds_traj, ds_id, ego_id, frm_id, lane='left'):
        '''
        返回指定帧，指定车道的车的id：
        1. ego车道就直接从Preceeding和Following获取
        2. left和right车道的话，就先选择一个最近的车辆作为mid车辆，然后从mid车辆的Preceeding和Following获取对应车道的前后车
        '''
        if lane == 'ego':
            # 直接从id对应帧的df中获取Preceeding和Following
            possible_pre_fol_vehc = ds_traj[ego_id][ds_traj[ego_id]['Frame_ID'] == frm_id][['Preceeding', 'Following']].values
            if possible_pre_fol_vehc.shape[0] == 0:
                return [0]
            tar_pre_vid, tar_fol_vid = possible_pre_fol_vehc[0]
            nbrs = [int(tar_fol_vid), int(tar_pre_vid)]  # put the ego at the first place
            return nbrs

        # 对应帧和id的y坐标以及lane 的id
        ego_y, ego_lane = ds_traj[ego_id][ds_traj[ego_id]['Frame_ID'] == frm_id][['Local_Y', 'Lane_ID']].values[0]
        ds_df = self.ds_df_dict[ds_id]

        #同一帧内左侧车道的id以及y 坐标
        if lane == 'left':
            vehs_tar_lane = ds_df[(ds_df['Frame_ID'] == frm_id) & (ds_df['Lane_ID'] == int(ego_lane - 1))][['Vehicle_ID', 'Local_Y']].values

        #同一帧内右侧车道的id以及y 坐标
        if lane == 'right':
            vehs_tar_lane = ds_df[(ds_df['Frame_ID'] == frm_id) & (ds_df['Lane_ID'] == int(ego_lane + 1))][['Vehicle_ID', 'Local_Y']].values
        if vehs_tar_lane.size == 0:
            return [0]

        # 把y坐标转变为与ego_y的绝对距离差，然后把距离差最小的作为中间车辆
        vehs_tar_lane[:, 1] = abs(vehs_tar_lane[:, 1] - ego_y)  # the longitudinal distance
        tar_mid_idx = np.where(vehs_tar_lane[:, 1] == np.min(vehs_tar_lane[:, 1]))[0][0]
        tar_mid_vid = vehs_tar_lane[:, 0][tar_mid_idx]
        tar_pre_vid, tar_fol_vid = ds_traj[tar_mid_vid][ds_traj[tar_mid_vid]['Frame_ID'] == frm_id][['Preceeding', 'Following']].values[0]
        nbrs = [int(tar_fol_vid), int(tar_mid_vid), int(tar_pre_vid)]
        return nbrs

    def get_nbrs_all(self, ds_id, ego_id, frm_id):
        '''
        得到指定id以及frame id的周围车辆：
        1. 如果是0的话，就删除，不加入
        2. 返回的nbrs_all,id为0就代表没有这辆车
        '''
        nbrs_all = []
        ds_traj = self.veh_id2traj_all[ds_id]
        for l in ['ego', 'left', 'right']:  # put the ego at the first place
            nbrs_1_lane = self.get_nbrs_one_lane(ds_traj, ds_id, ego_id, frm_id, lane=l)
            nbrs_1_lane = self.filter_nbrs(ds_traj, nbrs_1_lane, frm_id)
            nbrs_all += nbrs_1_lane
        # print(nbrs_all)
        # remove 0s from nbrs
        # nbrs_all = [vid for vid in nbrs_all if vid != 0]
        # print('nbrs without 0s {}'.format(nbrs_all))
        return nbrs_all

    def preprocess_data(self):
        '''
        保存每一个sample，顺序为：
        1. 数据集id
        2. 帧id
        3. tgt_vah 的id
        4. 是否变道,1为变道，0为不变道
        5之后的都是nbr
        '''
        samples = []
        # 遍历每个数据集
        for ds_id, ds_traj in self.veh_id2traj_all.items():
            print(f"preprocessing data for dataset {ds_id}")
            ds_samples = []
            # 遍历每个tgt_id
            for tgt_id, tgt_traj in tqdm(ds_traj.items()):
                # 存活帧数小于81帧的就跳过
                if len(tgt_traj) < 81:
                    continue
                # 每一帧都可以作为一个样本，从第31帧开始，到倒数50帧
                frm_ids = tgt_traj['Frame_ID'].to_numpy()
                for ref_frm_id in tqdm(frm_ids[31:-50], leave=False):
                    # 得到指定帧，指定id的轨迹
                    traj = self.get_traj(tgt_traj, ref_frm_id)
                    # 判断是否有变道
                    lc = self.is_lc(traj)

                    # 得到nbrs
                    nbrs = self.get_nbrs_all(ds_id, tgt_id, ref_frm_id)
                    sample = [ds_id, ref_frm_id, tgt_id, lc] + nbrs
                    ds_samples.append(sample)
                    # break
                # break
            with open(f"ngsim_samples_list/ngsim_samples_ds{ds_id}", "wb") as fp:  #Pickling
                pickle.dump(ds_samples, fp)
            samples.extend(ds_samples)
            # break

        with open(f"ngsim_samples_list/ngsim_samples", "wb") as fp:  #Pickling
            pickle.dump(samples, fp)

        print(f"total samples: {len(samples)}")


if __name__ == '__main__':

    # ngsim_samples_pre = NgsimSamplesPre()
    # ngsim_samples_pre.preprocess_data()

    with open("ngsim_samples_list/ngsim_samples", "rb") as fp:  # Unpickling
        samples_list = pickle.load(fp)
    lc_samples = [i for i in samples_list if i[3] == 1]
    no_lc_samples = [i for i in samples_list if i[3] == 0]

    print(f"The number of lane change scenario: {len(lc_samples)}")
    print(f"The number of no lane change scenario: {len(no_lc_samples)}")
    print(f"The ratio of lc to no_lc: {len(lc_samples)/len(no_lc_samples)}")

    import random
    rd_lc_samples = random.sample(lc_samples, 10)
