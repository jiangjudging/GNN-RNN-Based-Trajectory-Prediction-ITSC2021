import os
import numpy as np
import pandas as pd
import pickle
import time
import math


class once_LC_veh_selector():

    def __init__(self, csv_data_path='/home/xy/trajectories-0805am-0820am.csv', t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        print('\n working on {} \n'.format(csv_data_path))
        self.csv_data_path = csv_data_path
        self.df_0 = pd.read_csv(self.csv_data_path)
        self.veh_id2traj_all = self.get_veh_id2traj_all()

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
        # 选出在要求区间的frame
        vehposs = df0[(df0['Frame_ID'] >= frm_stpt) & (df0['Frame_ID'] < frm_enpt)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc']]
        return vehposs

    def find_lc_frame(self, traj_i):
        '''
        找到变道的那一帧，返回帧id以及原始车道和换道后的车道
        '''
        traji_laneid = traj_i['Lane_ID'].values
        traji_frameid = traj_i['Frame_ID'].values

        lc_idx = np.nonzero(traji_laneid[1:] - traji_laneid[:-1])
        lc_idx = lc_idx[0]  #因为np.nonzero默认返回的为tuple，所以要先取第0个
        if len(lc_idx) != 1:  #只允许有一次变道，否则返回0，0，0
            return 0, 0, 0
        else:
            lc_idx = lc_idx[0]
            lc_frame = traji_frameid[lc_idx]
            o_lane = traji_laneid[lc_idx]
            t_lane = traji_laneid[lc_idx + 1]
            return lc_frame, o_lane, t_lane

    def cal_LC_nums(self, lane_IDs):
        '''
        统计某个id的变道次数
        '''

        lc_num = 0
        for i in range(len(lane_IDs) - 1):
            last_laneID = lane_IDs[i]
            cur_laneID = lane_IDs[i + 1]
            if last_laneID != cur_laneID:
                lc_num += 1
        return lc_num

    def num_LC_by_veh(self):
        '''
        统计满足以下条件的veh id:
        1. 没有出现在过lane id大于6的路段
        2. 只变道过1次
        3. 生命周期内，y方向的跨度要大于1000 feet
        4. 变道的y方向的位置要在300-1900之间
        5. 变道前后60帧，x方向的变化大于10 feet
        '''
        LC_by_veh = {}
        vehs = []
        for vi, traji in self.veh_id2traj_all.items():
            lc_num_i = self.cal_LC_nums(traji['Lane_ID'].values)
            if np.max(traji['Lane_ID'].values) > 6:  #如果曾经出现在了lane id大于6的路段，就不要他
                continue
            if lc_num_i == 1:
                y_max = np.max(self.veh_id2traj_all[vi]['Local_Y'].values)
                y_min = np.min(self.veh_id2traj_all[vi]['Local_Y'].values)
                if y_max - y_min > 1000:  # 跨度要大于1000 feet
                    # find the LC frame
                    lc_frm, ol, tl = self.find_lc_frame(self.veh_id2traj_all[vi])
                    if lc_frm == 0:  # 一开始就变道的话，也不要
                        continue
                    long_pos = self.veh_id2traj_all[vi][(self.veh_id2traj_all[vi]['Frame_ID'] == lc_frm)]['Local_Y'].values[0]
                    if long_pos < 300 or long_pos > 1900:  #变道的区间要在300-1900之间
                        continue
                    LC_clips = self.get_frames(self.veh_id2traj_all[vi], frm_stpt=lc_frm - 60, frm_enpt=lc_frm + 60)  # 取变道前后60帧
                    befor_LC_clips = self.get_frames(self.veh_id2traj_all[vi], frm_stpt=lc_frm - 60, frm_enpt=lc_frm)
                    after_LC_clips = self.get_frames(self.veh_id2traj_all[vi], frm_stpt=lc_frm + 1, frm_enpt=lc_frm + 60)
                    x_max = np.max(LC_clips['Local_X'].values)
                    x_min = np.min(LC_clips['Local_X'].values)
                    if x_max - x_min > 10:  # 如果x方向的变化大于10 feet的话，就把id加进去
                        vehs.append(vi)
        print('there are {} vehicles in {} data changed lane once only.'.format(len(vehs), self.csv_data_path))
        return vehs


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    csv_data_path_list = ['data_i80/trajectories-0400pm-0415pm.csv', 'data_i80/trajectories-0500pm-0515pm.csv', 'data_i80/trajectories-0515pm-0530pm.csv']
    # csv_data_path_list = ['data2/trajectories-0750am-0805am.csv', 'data2/trajectories-0805am-0820am.csv', 'data2/trajectories-0820am-0835am.csv']
    once_lc_vehs_dict = {}

    for data_path in csv_data_path_list:
        lc_v_slector = once_LC_veh_selector(csv_data_path=data_path)
        once_lc_veh_ids = lc_v_slector.num_LC_by_veh()

        data_k = data_path.split('trajectories-')[1].split('.')[0]
        once_lc_vehs_dict[data_k] = once_lc_veh_ids  #key为data_k，value为选择出来的id

    # save the selected vehicle IDs
    save_obj(once_lc_vehs_dict, 'once_LC_vehs_i80')

    print(once_lc_vehs_dict)