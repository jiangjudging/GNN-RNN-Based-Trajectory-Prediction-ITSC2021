import os
import os.path as osp
import time
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from plot_helper import find_files
import pickle, json
from plot_instan import rotate_points, plot_glb_frm


class LD_Rotate_Dataset(Dataset):

    def __init__(self, samples_list, ds_df_dict, t_h=30, t_f=50):
        super().__init__()
        # Initialization
        self.samples_list = samples_list
        self.ds_df_dict = ds_df_dict

        self.t_h = t_h
        self.t_f = t_f

        self.veh_id2traj_all = self.get_veh_id2traj_all()

        print(f'there are {len(self)} data pieces')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample = self.samples_list[index]
        ds_id = sample[0]
        frm_id = sample[1]
        tgt_id = sample[2]
        nbrs_id = sample[3:]

        hist_all, f1 = self.get_data_item(ds_id, tgt_id, frm_id, nbrs_id)
        data_item = self.turn_data_item_to_pyg(hist_all, f1)
        data_item.ds_id = ds_id
        data_item.frm_id = frm_id
        data_item.tgt_id = tgt_id
        # data_item = torch.load(ID)
        # # data_item.node_feature = data_item.node_feature.float()
        # data_item.x = data_item.node_feature.float()
        # data_item.y = data_item.y.float()

        # data_item.fname = ID.split('/')[-1][:-4]

        # print(data_item)
        return data_item

    def get_veh_id2traj_all(self):
        '''
        得到所有数据集以及其所有obj id的一个轨迹
        返回一个字典，key为数据集id，value为字典，value字典的key为obj id, value为该id的轨迹
        '''
        print('getting trajectories of vehicles...')
        veh_id2traj = {}
        for ds_id, df in self.ds_df_dict.items():
            veh_id2traj[ds_id] = {}
            veh_IDs = set(df['obj_id'].values)

            for vi in veh_IDs:
                vi_traj = df[(df['obj_id'] == vi)][['frame_id', 'obj_id', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
                veh_id2traj[ds_id][vi] = vi_traj
            print('totally {} vehicles in dataset {}'.format(len(veh_id2traj[ds_id]), ds_id))
        return veh_id2traj

    def get_data_item(self, ds_id, ego_id, frm_id, nbrs_id):
        '''
        得到指定ego_id在指定帧的数据，包括目标车辆的历史，未来轨迹，以及邻居车辆的历史轨迹
        1. hist_all，第一位就是ego车辆的历史轨迹，后面就是邻居车辆的历史轨迹
        2. f1就是ego车辆的未来轨迹
        3. 邻居车辆确保在30帧内都存在
        '''
        # 得到ego车辆的ref_pos,ego_loc_pos,以及他的历史轨迹ego_hist
        ref_pos, ego_loc_pos, ego_hist = self.get_ego_hist(ds_id, ego_id, frm_id)
        yaw = self.get_ego_yaw(ds_id, frm_id)
        if ref_pos is None:
            return np.empty([0, 31, 2]), np.empty([0, 50, 2]), []

        # 得到周围车辆的历史轨迹
        nbrs_hist = self.get_nbrs_hist(ds_id, nbrs_id, frm_id, ref_pos)
        # print(nbrs_all)

        # initialize data
        hist_all = np.zeros([len(nbrs_hist) + 1, 31, 2])
        # hist_all[0] = ego_hist  # 第一个必须是ego历史轨迹
        hist_all[0] = rotate_points(ego_hist, -yaw)  # 第一个必须是ego历史轨迹-->旋转后的
        f1 = np.zeros([1, 50, 2])

        # get the fut of ego (target)
        f = self.get_fut(ds_id, ego_id, frm_id, ref_pos)
        f = rotate_points(f, -yaw)
        if len(f) == 0:
            return np.empty([0, 31, 2]), np.empty([0, 50, 2]), []
        f1[0] = f

        # get hist of all vehicles (nbrs and ego)
        for i, nbr_hist in enumerate(nbrs_hist):
            # hist_all[i + 1] = nbr_hist
            hist_all[i + 1] = rotate_points(nbr_hist, -yaw)

        # edges of a star-like graph

        return hist_all, f1

    def turn_data_item_to_pyg(self, hist_all, fut_gt):
        '''
        建图：
        1. 以所有车辆作为节点，他们的历史轨迹作为节点特征
        2. 建立星型的边，ego在中心，与邻居车辆有个双向的边，然后与ego自身也有个边(self loop)
        '''
        # node features and the ground truth future trajectory
        Node_f = torch.from_numpy(hist_all).float()  # node features
        fut_gt = torch.from_numpy(fut_gt).float()  # the label, ground truth of future traj

        # edges, star-like with self loop
        edge_st_pt = torch.tensor([i for i in range(hist_all.shape[0])]).unsqueeze(0).long()
        edge_nd_pt = torch.zeros(hist_all.shape[0]).unsqueeze(0).long()
        in_edges = torch.cat((edge_st_pt, edge_nd_pt), dim=1)  # tensor([[0, 1, 2, 3, 0, 0, 0, 0]])
        out_edges = torch.cat((edge_nd_pt, edge_st_pt), dim=1)  # tensor([[0, 0, 0, 0, 0, 1, 2, 3]])
        # tensor([[0, 1, 2, 3, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 2, 3]])
        Edges = torch.cat((in_edges, out_edges), dim=0)

        # turn to pyg data
        one_pyg_data = Data(x=Node_f, edge_index=Edges, y=fut_gt)

        return one_pyg_data

    def get_frames(self, df0, frm_stpt=12, frm_enpt=610):
        '''
        选出在要求区间的frame,左闭右开原则
        '''
        vehposs = df0[(df0['frame_id'] >= frm_stpt) &
                      (df0['frame_id'] < frm_enpt)][['frame_id', 'obj_id', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
        return vehposs

    def get_ego_yaw(self, ds_id, frm_id):
        '''
        得到指定frm_id的主车的yaw角
        '''
        ego_frame_df = self.veh_id2traj_all[ds_id][0][self.veh_id2traj_all[ds_id][0]['frame_id'] == frm_id]
        yaw = ego_frame_df['oritentation_yaw'].values[0]
        return yaw

    def get_ego_hist(self, ds_id, ego_id, frm_id):
        '''
        返回ego车辆在frm_id的坐标，作为ref_pos，orie_w，以及历史hist_len帧的位置（相对于ref_pos）
        '''
        if ego_id not in self.veh_id2traj_all[ds_id].keys():  # 如果不在之前找到的id内的话，就返回np.empty([0, 2])
            return None, None, np.empty([0, 2])
        # 先计算ref_pos以及 orie_w
        ego_frame_df = self.veh_id2traj_all[ds_id][ego_id][self.veh_id2traj_all[ds_id][ego_id]['frame_id'] == frm_id]
        ref_pos = ego_frame_df[['GlobalX', 'GlobalY']].values[0]
        ego_loc_pos = ego_frame_df[['LocalX', 'LocalY']].values[0]

        # 得到历史轨迹
        veh_track = self.get_frames(self.veh_id2traj_all[ds_id][ego_id], frm_stpt=frm_id - self.t_h, frm_enpt=frm_id + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values

        # 减去ref_pos，得到相对于ref_pos的位置
        veh_track = veh_track - ref_pos

        # 如果得到的帧数小于hist_len + 1，也返回empty
        if len(veh_track) < self.t_h + 1:
            return None, None, np.empty([0, 2])
        return ref_pos, ego_loc_pos, veh_track

    def get_fut(self, ds_id, veh_id, frm_id, ref_pos):
        '''
        返回veh_id的fut位置：
        1. 是从frm_id+1开始的50帧，左闭右开
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于50帧的话，返回np.empty([0, 2])
        '''
        veh_track = self.get_frames(self.veh_id2traj_all[ds_id][veh_id], frm_stpt=frm_id + 1, frm_enpt=frm_id + self.t_f + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < self.t_f:
            return np.empty([0, 2])
        return veh_track

    def get_nbr_hist(self, ds_id, veh_id, frm_id, ref_pos):
        '''
        返回veh_id的hist位置：
        1. 是从frm_id-30开始的31帧，左闭右开    ？？？ 这里为什么要31帧，也是不解
        2. 把位置统一减去参考位置，也就是以参考位置为原点(0,0)
        3. 如果track的长度小于31帧的话，返回np.empty([0, 2])
        '''
        if veh_id not in self.veh_id2traj_all[ds_id].keys():  # 如果不在之前找到的id内的话，就返回np.empty([0, 2])
            return np.empty([0, 2])

        # ref_pos = self.veh_id2traj_all[ego_id][self.veh_id2traj_all[ego_id]['frame_id'] == frm_id][['GlobalX', 'GlobalY']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_all[ds_id][veh_id], frm_stpt=frm_id - self.t_h, frm_enpt=frm_id + 1)
        veh_track = veh_track[['GlobalX', 'GlobalY']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < self.t_h + 1:
            return np.empty([0, 2])
        return veh_track

    def get_nbrs_hist(self, ds_id, nbrs_id, frm_id, ref_pos):
        nbrs_hist = []
        for nbr_id in nbrs_id:
            veh_track = self.get_nbr_hist(ds_id, nbr_id, frm_id, ref_pos)
            nbrs_hist.append(veh_track)
        return nbrs_hist


def get_ds_id_dict():
    ds_id_map = "dataset_id_mapping.json"
    ds_id_dict = {}

    with open(ds_id_map, "r") as outfile:
        ds_id_dict = json.load(outfile)
    return ds_id_dict


def get_smp_list(sample_dir, prefix):
    samples_list = []
    samples = find_files(sample_dir, prefix=prefix, suffix="")
    for sample in samples:
        print(sample)
        with open(sample, 'rb') as fp:  # Unpickling
            temp = pickle.load(fp)
            samples_list += temp
    print(f"total samples count: {len(samples_list)}")
    return samples_list


def get_ds_df_dict(csv_dir, ds_id_dict):
    csvs = find_files(csv_dir, suffix='.csv')
    ds_df_dict = {}
    for csv in csvs:
        df = pd.read_csv(csv)
        id = -1
        for ds_id, ds_name in ds_id_dict.items():
            csv_name = csv.split('/')[-1]
            if csv_name.startswith(ds_name):
                id = int(ds_id)
                break
        if id == -1 or id in ds_df_dict:
            raise ValueError(f"wrong csv name : {csv}; id: {id}")
        ds_df_dict[id] = df

    print(f"total dataset count: {len(ds_df_dict)}")
    return ds_df_dict


def get_smp_list_ds_df_dict(sample_dir, csv_dir, prefix="LIDAR"):
    sample_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + sample_dir
    csv_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + csv_dir

    # 获得数据集id mapping
    ds_id_dict = get_ds_id_dict()

    # 获得所有样本
    samples_list = get_smp_list(sample_dir, prefix)

    # 获得所有df
    ds_df_dict = get_ds_df_dict(csv_dir, ds_id_dict)

    return samples_list, ds_df_dict


def get_tgt_smp(samples_list, ds_id, tgt_id):
    # 返回知道数据集和id的样本list
    ret = [smp for smp in samples_list if smp[0] == ds_id and smp[2] == tgt_id]

    return ret


if __name__ == '__main__':
    # import pickle
    # with open("ld_data/processed_samples_list/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060", "rb") as fp:  # Unpickling
    #     samples_list = pickle.load(fp)

    # csv_path1 = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP_fusion_10hz.csv"
    # ds_df1 = pd.read_csv(csv_path1)
    # ds_df_dict = {1: ds_df1}

    # ld_dataset = LD_Rotate_Dataset(samples_list, ds_df_dict)
    # print(len(ld_dataset))

    import random
    random.seed(1)
    # idx_lists = random.sample(range(0, len(ld_dataset)), 10)
    # print(idx_lists)

    # for idx in idx_lists:
    #     print(samples_list[idx])
    #     print(ld_dataset[idx].x)
    #     print(ld_dataset[idx].y)
    #     plot_glb_frm(ld_dataset[idx].x, ld_dataset[idx].y, 0, '0')

    from ld_data_pre import ibeo_csv_dict, ibeo_csv_dir
    sample_dir = "ld_data/processed_ibeo_samples_list"
    samples_list = get_smp_list(sample_dir, prefix="")
    ds_df_dict = get_ds_df_dict(ibeo_csv_dir, ibeo_csv_dict)
    ld_ibeo_dataset = LD_Rotate_Dataset(samples_list, ds_df_dict)
    print(len(ld_ibeo_dataset))

    idx_lists = random.sample(range(0, len(ld_ibeo_dataset)), 10)
    print(idx_lists)

    for idx in idx_lists:
        print(samples_list[idx])
        print(ld_ibeo_dataset[idx].x)
        print(ld_ibeo_dataset[idx].y)
        plot_glb_frm(ld_ibeo_dataset[idx].x, ld_ibeo_dataset[idx].y, 0, '0')
