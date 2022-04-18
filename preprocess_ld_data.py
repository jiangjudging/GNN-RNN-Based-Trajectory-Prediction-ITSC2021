import torch, shutil, random, os, subprocess, logging, sys
import os.path as osp
import numpy as np
import pandas as pd
from plot_helper import find_files, create_new_dir
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from plot_helper import find_files

random.seed(0)

#配置logging
cur_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/logging"
datetime_format = '%Y_%m_%d_%H_%M'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(f'{cur_path}/{datetime.now():{datetime_format}}_logging.log'),
                              logging.StreamHandler()])

bag2csv_path = "/home/jiang/VW/toolchain/compass_script/tools/bag2csv"


def move_bag_to_dir(bag_dirpath):
    bags = find_files(bag_dirpath)
    for b in bags:
        bag_name = b.split('/')[-1][:-4]
        dst_dir = create_new_dir(bag_dirpath, bag_name)
        dst_path = osp.join(dst_dir, b.split('/')[-1])
        shutil.move(b, dst_path)


def run_bag2csv(_cwd, bag_path, output_dir):
    subprocess.check_output(['./bag2csv', bag_path, output_dir], cwd=_cwd, stderr=subprocess.STDOUT)


def run_bag2csv_all(_cwd, bag_dir):
    bags = find_files(bag_dir, recursive=True)
    for b in tqdm(bags):
        out_path = osp.dirname(b)
        run_bag2csv(_cwd, b, out_path)


def del_obj_out_range(df, min_dist=0, max_dist=90):
    '''
    删除最远距离小于min_dist或者最小距离大于max_dist的obj
    '''
    # src_obj_id = df['obj_id'].unique()
    src_obj_id = df['object.id'].unique()
    ret_df = pd.DataFrame()

    for obj_id in src_obj_id:
        # obj_id_df = df[df.obj_id == obj_id]
        # _min_dist = min(obj_id_df['LocalX'])
        # _max_dist = max(obj_id_df['LocalX'])

        obj_id_df = df[df['object.id'] == obj_id]
        _min_dist = min(obj_id_df['pose.position.x'])
        _max_dist = max(obj_id_df['pose.position.x'])

        if _min_dist > max_dist or _max_dist < min_dist:
            logging.info(f"delete: The obj id: {obj_id:4} has min_dist {_min_dist:6,.2f} and max_dist {_max_dist:6,.2f}")
        else:
            ret_df = ret_df.append(obj_id_df)
    ret_df.sort_values(by=['object list No.', 'object.id'], inplace=True)
    return ret_df


def del_obj_life_less(df, duration=3):
    '''
    删除生命周期短于duration的obj
    '''
    # src_obj_id = df['obj_id'].unique()
    src_obj_id = df['object.id'].unique()
    ret_df = pd.DataFrame()

    for obj_id in src_obj_id:
        # obj_id_df = df[df.obj_id == obj_id]
        obj_id_df = df[df['object.id'] == obj_id]
        # _duration = max(obj_id_df['timestamp']) - min(obj_id_df['timestamp'])
        _duration = max(obj_id_df['bag timestamp']) - min(obj_id_df['bag timestamp'])
        if _duration < duration:
            logging.info(f"delete: The obj id: {obj_id:4} has duration {_duration:6,.2f}s.")
        else:
            ret_df = ret_df.append(obj_id_df)
    # ret_df.sort_values(by=['frame_id','obj_id'],inplace=True)
    ret_df.sort_values(by=['object list No.', 'object.id'], inplace=True)
    return ret_df


def del_obj_dim_less(df, dx=1.5, dy=0.8):
    '''
    删除dimension_x < 1.5m 或者dimension_y < 0.8m的obj
    '''
    # src_obj_id = df['obj_id'].unique()
    src_obj_id = df['object.id'].unique()
    ret_df = pd.DataFrame()

    for obj_id in src_obj_id:
        # obj_id_df = df[df.obj_id == obj_id]
        obj_id_df = df[df['object.id'] == obj_id]
        # _duration = max(obj_id_df['timestamp']) - min(obj_id_df['timestamp'])
        x = np.mean(obj_id_df['dimension_x'])
        y = np.mean(obj_id_df['dimension_y'])
        if x < dx or y < dy:
            logging.info(f"delete: The obj id: {obj_id:4} has dx {x:6,.2f}(<{dx}) or dy {y:6,.2f}(<{dy}) .")
        else:
            ret_df = ret_df.append(obj_id_df)
    # ret_df.sort_values(by=['frame_id','obj_id'],inplace=True)
    ret_df.sort_values(by=['object list No.', 'object.id'], inplace=True)
    return ret_df


def check_dup_id_one_frm(df):
    # frm_id_obj_id_df = df[['object list No.', 'object.id']].copy()
    frm_id_obj_id_df = df[['frame_id', 'obj_id']].copy()
    # repeat_df = frm_id_obj_id_df[frm_id_obj_id_df.duplicated()]
    # for idx, row in repeat_df.iterrows():
    #     # print(row)
    #     logging.info(f"The obj id: {row['object.id']:4} has repeat occurance at  {row['object list No.']} frame.")
    df = frm_id_obj_id_df.groupby(frm_id_obj_id_df.columns.tolist()).size().reset_index().rename(columns={0: 'records'})
    df = df[df['records'] > 1]
    for idx, row in df.iterrows():
        # print(row)
        # logging.info(f"The obj id: {row['object.id']:4} has repeat {row['records']} times at  {row['object list No.']} frame.")
        logging.info(f"The obj id: {row['obj_id']:4} has repeat {row['records']} times at  {row['frame_id']} frame.")

    return df


def dup_id_check(df, gap=0.07):
    '''
    检查是否存在id的间隔大于gap的值
    '''
    # src_obj_id = df['obj_id'].unique()
    src_obj_id = df['object.id'].unique()
    ret_df = pd.DataFrame()
    large_gap = defaultdict(list)
    for obj_id in src_obj_id:
        # obj_id_df = df[df.obj_id == obj_id]
        # ts = obj_id_df['timestamp'].to_numpy()
        # _date = obj_id_df['date'].to_numpy()
        obj_id_df = df[df['object.id'] == obj_id]
        ts = obj_id_df['bag timestamp'].to_numpy()
        f = obj_id_df['object list No.'].to_numpy()
        # _date = obj_id_df['date'].to_numpy()
        _gap = ts[1:] - ts[:-1]
        idxs = np.argwhere(_gap > gap)
        if len(idxs) > 0:
            for i in idxs:
                t1 = ts[i[0]]
                t2 = ts[i[0] + 1]
                f1, f2 = f[i[0]], f[i[0] + 1]
                large_gap[obj_id].append([f1, f2])
                logging.info(f"The obj id: {obj_id:4} has gap of {t2-t1:5,.3f}s ({f2-f1:2} frame) at  {datetime.utcfromtimestamp(t1)} ({t1:.3f}).")
    return large_gap


def linear_interpol(df, gap_dict):
    '''
    把gap值较大的id的obj进行线性插值
    '''
    for obj_id, gap_list in gap_dict.items():
        obj_id_df = df[df['object.id'] == obj_id]
        for gap in gap_list:
            start_frame_id, end_frame_id = gap[0], gap[1]
            gap_frames = end_frame_id - start_frame_id - 1

            start_frame, end_frame = obj_id_df[obj_id_df['object list No.'] == start_frame_id], obj_id_df[obj_id_df['object list No.'] == end_frame_id]
            start_x, end_x = start_frame['pose.position.x'], end_frame['pose.position.x']
            start_y, end_y = start_frame['pose.position.y'], end_frame['pose.position.y']
            start_t, end_t = start_frame['bag timestamp'], end_frame['bag timestamp']

            x = np.linspace(start_x, end_x, num=gap_frames + 2)[1:-1]
            # print(x)
            y = np.linspace(start_y, end_y, num=gap_frames + 2)[1:-1]
            t = np.linspace(start_t, end_t, num=gap_frames + 2)[1:-1]
            gap_frame_ids = [i for i in range(start_frame_id + 1, end_frame_id)]
            for _x, _y, _t, frame_id in zip(x, y, t, gap_frame_ids):
                new_frame = start_frame.copy()
                # print(new_frame)
                # print(_x)
                new_frame['pose.position.x'] = _x
                new_frame['pose.position.y'] = _y
                new_frame['object list No.'] = frame_id
                new_frame['bag timestamp'] = _t
                new_frame['header timestamp'] = _t
                df = df.append(new_frame)
    df.sort_values(by=['object list No.', 'object.id'], inplace=True)
    return df


def del_obj_id(df, to_del_id):
    ret_df = df[~df['object.id'].isin(to_del_id)]
    return ret_df


def get_rota_mat(a):
    ''' 
    返回左乘的旋转矩阵,a是逆时针旋转的弧度
    '''
    rota_mat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return rota_mat


def rotate_points(arr, a):
    ''' 
    把arr逆时针旋转a弧度
    arr是[n*2]的形状
    返回的结果也是n*2的结果
    '''
    rm = get_rota_mat(a)
    arr2 = np.dot(arr, rm.T)
    return arr2


def cal_globalXY(merge_df):
    '''
    Calculate global X and Y
    '''
    frame_id_set = merge_df['frame_id'].unique()
    for frame_id in tqdm(frame_id_set):
        frame_id_mask = (merge_df['frame_id'] == frame_id)
        obj_id_mask = merge_df['obj_id'] != 0

        ego_global_x, ego_global_y, ego_yaw = merge_df[frame_id_mask].iloc[0]['GlobalX'], merge_df[frame_id_mask].iloc[0]['GlobalY'], merge_df[
            frame_id_mask].iloc[0]['oritentation_yaw']

        #先把局部坐标，按照主车的yaw角进行旋转
        obj_local_pos = merge_df.loc[frame_id_mask & obj_id_mask, ['LocalX', 'LocalY']].to_numpy()
        obj_local_pos_rot = rotate_points(obj_local_pos, ego_yaw)

        # 旋转后再进行位移
        merge_df.loc[frame_id_mask & obj_id_mask, ['GlobalX']] = obj_local_pos_rot[:, 0] + ego_global_x
        merge_df.loc[frame_id_mask & obj_id_mask, ['GlobalY']] = obj_local_pos_rot[:, 1] + ego_global_y

    # merge_df.loc[:, 'GlobalX'] += merge_df['LocalX']
    # merge_df.loc[:, 'GlobalY'] += merge_df['LocalY']
    return merge_df


def merge_obj_ego(obj_df, ego_df):
    '''
    把obj_df以及ego_df合并起来,并且算出GlobalX,以及GlobalY
    '''

    # 取出需要的列
    obj_df = obj_df[['object list No.', 'bag timestamp', 'object.id', 'class_label_pred', 'pose.position.x', 'pose.position.y', 'pose.orientation.w']].copy()
    ego_df = ego_df[['frame seq', 'bag timestamp', 'position x', 'position y', 'oritentation yaw']].copy()

    # 给ego_df的列重命名以及赋初值
    ego_df.rename(columns={
        'frame seq': 'frame_id',
        'bag timestamp': 'timestamp',
        'position x': 'GlobalX',
        'position y': 'GlobalY',
        'oritentation yaw': 'oritentation_yaw'
    },
                  inplace=True)
    ego_df['class'] = 'Car'
    ego_df.loc[:, ['LocalX', 'LocalY', 'obj_id']] = 0

    # 给obj_df的列重命名以及赋初值
    obj_df.rename(columns={
        'object list No.': 'frame_id',
        'bag timestamp': 'timestamp',
        'object.id': 'obj_id',
        'class_label_pred': 'class',
        'pose.position.x': 'LocalX',
        'pose.position.y': 'LocalY',
        'pose.orientation.w': 'oritentation_yaw'
    },
                  inplace=True)
    obj_df.loc[:, ['GlobalX', 'GlobalY']] = 0

    # 把两个df concate起来,并且按照frame_id和obj_id排序,列重新排列,增加一个可读时间
    merge_df = pd.concat([ego_df, obj_df])
    merge_df.sort_values(by=['frame_id', 'obj_id'], inplace=True)
    # merge_df = merge_df[['frame_id', 'timestamp', 'obj_id', 'class', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
    # merge_df['date'] = pd.to_datetime(merge_df['timestamp'],unit='s').dt.tz_localize('Europe/Berlin')
    merge_df['date'] = pd.to_datetime(merge_df['timestamp'], unit='s')
    merge_df = merge_df[['frame_id', 'date', 'timestamp', 'obj_id', 'class', 'LocalX', 'LocalY', 'GlobalX', 'GlobalY', 'oritentation_yaw']]
    # 计算golbal x y
    merge_df = cal_globalXY(merge_df)
    return merge_df


def down_sample(df, r=2):
    src_frame_id = df['frame_id'].unique()
    dst_frame_id = np.arange(start=0, stop=len(src_frame_id), step=r)

    ret_df = pd.DataFrame()
    for idx, frame_id in enumerate(dst_frame_id):
        frame_id_df = df[df['frame_id'] == frame_id].copy()
        frame_id_df.loc[:, 'frame_id'] = idx
        ret_df = ret_df.append(frame_id_df)
    return ret_df


def get_save_path(file_path, new_suf, del_suf=".csv"):
    out_dir = os.path.dirname(file_path)
    del_suf_len = len(del_suf)
    out_name = file_path.split('/')[-1][:-del_suf_len] + new_suf
    out_path = os.path.join(out_dir, out_name)
    return out_path


def cut_df(df, cut_time_periods):
    ret_dfs = []
    for cut_time_period in cut_time_periods:
        start = cut_time_period[0] if cut_time_period[0] is not None else df['timestamp'].min()
        end = cut_time_period[1] if cut_time_period[1] is not None else df['timestamp'].max()
        print(start)
        print(end)
        ret_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        ret_dfs.append(ret_df)
    return ret_dfs


if __name__ == '__main__':
    try:
        # bag_path = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20220126_134418_G310-PDX-007-002-001_000000-000036_LD_final_OD_MERGE_OPP_LDE/LIDAR_LJ02766_20220126_134418_G310-PDX-007-002-001_000000-000036_LD_final_OD_MERGE_OPP_LDE.bag"
        # out_dir = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20220126_134418_G310-PDX-007-002-001_000000-000036_LD_final_OD_MERGE_OPP_LDE"
        # run_bag2csv(bag2csv_path, bag_path, out_dir)
        # exit()
        dataset_dirpath = '/home/jiang/trajectory_pred/ld_dataset/ibeo_dataset'

        # 找出所有的obj,ego的csv
        obj_csvs = find_files(dataset_dirpath, recursive=True, suffix='obj.csv')
        ego_csvs = find_files(dataset_dirpath, recursive=True, suffix='ego.csv')
        # logging.info(len(obj_csvs), obj_csvs)
        # logging.info(len(ego_csvs), ego_csvs)
        '''

        # 过滤每一个obj_csv,保存过滤后的csv
        # 1.删除没有在0-90m出现过的obj
        min_dist, max_dist = -40, 90
        # 2.删除生命周期小于3s的obj
        duration = 3
        # 3.删除dimension_x < 1.5m 或者dimension_y < 0.8m
        dx, dy = 1.5, 0.8
        # 4.一帧出现两个相同的id TODO
        obj_filter_dfs = []
        for obj_csv in obj_csvs:
            logging.info(obj_csv)
            obj_df = pd.read_csv(obj_csv)
            del1_obj_df = del_obj_out_range(obj_df, min_dist=min_dist, max_dist=max_dist)
            del2_obj_df = del_obj_life_less(del1_obj_df, duration=duration)
            del3_obj_df = del_obj_dim_less(del2_obj_df, dx=dx, dy=dy)
            out_path = get_save_path(obj_csv, "_filter.csv")
            logging.info(out_path)
            logging.info("\n\n")
            del3_obj_df.to_csv(out_path, index=False)
            obj_filter_dfs.append(del3_obj_df)
                
        
        # dataset_dirpath = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210916_052044_G260-PDX-006-001-052_000003-000060_LD_final_OD_MERGE_OPP"
        obj_filter_csvs = find_files(dataset_dirpath, recursive=True, suffix='obj_filter.csv')
        # gap_dict_list = []
        # 进行插值
        for obj_filter_csv in obj_filter_csvs:
            logging.info(obj_filter_csv)
            obj_filter_df = pd.read_csv(obj_filter_csv)
            gap_dict = dup_id_check(obj_filter_df, gap=0.2)
            # obj_filter_interp_df = linear_interpol(obj_filter_df, gap_dict)
            # out_path = get_save_path(obj_filter_csv, '_interp2.csv')
            # obj_filter_interp_df.to_csv(out_path, index=False)
            # gap_dict_list.append(gap_dict)
        '''
        #生成视频进行数据检查,看有没有需要删除的id

        #删除视频检查以后需要清除的id
        # interp_obj_csv = "/home/jiang/trajectory_pred/ld_dataset/ibeo_dataset/OD_LiangDao_20220318_9988_164021/OD_LiangDao_20220318_9988_164021_obj_filter.csv"
        # interp_obj_df = pd.read_csv(interp_obj_csv)
        # to_del_id = [75]
        # del_obj_id_df = del_obj_id(interp_obj_df, to_del_id)
        # out_path = get_save_path(interp_obj_csv, '_del.csv')
        # del_obj_id_df.to_csv(out_path, index=False)

        #把obj_df以及ego_df合并起来，并且算出GlobalX，以及GlobalY
        # ego_csvs = find_files(dataset_dirpath, recursive=True, suffix='ego.csv')
        # interp2_csvs = find_files(dataset_dirpath, recursive=True, suffix='del.csv')
        # # final_obj_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP_obj_filter_interp_del.csv"
        # # final_ego_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP_ego.csv"
        # assert len(ego_csvs) == len(interp2_csvs), "文件数量不符"
        # for ego_csv, interp2_csv in zip(ego_csvs, interp2_csvs):
        #     logging.info(ego_csv.split('/')[-1])
        #     assert ego_csv.split('/')[-1][:32] == interp2_csv.split('/')[-1][:32], "文件不对应"
        #     final_obj_df = pd.read_csv(interp2_csv)
        #     final_ego_df = pd.read_csv(ego_csv)

        #     merge_df = merge_obj_ego(final_obj_df, final_ego_df)
        #     merge_out_path = get_save_path(ego_csv, "_fusion.csv", "_ego.csv")
        #     merge_df.to_csv(merge_out_path, index=False)

        # 把20hz的数据降到10hz
        # fusion_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_101959_G260-PDX-006-001-052_000000-000060_LD_final_OD_MERGE_OPP_fusion.csv"
        # fusion_df = pd.read_csv(fusion_csv)
        # fusion_10hz_df = down_sample(merge_df)
        # fusion_10hz_out_path = get_save_path(ego_csv, "_fusion_10hz.csv", "_ego.csv")
        # fusion_10hz_df.to_csv(fusion_10hz_out_path, index=False)

        # 剪切时间段
        fusion_csv = "/home/jiang/trajectory_pred/ld_dataset/ibeo_dataset/OD_LiangDao_20220318_9988_171225/OD_LiangDao_20220318_9988_171225_fusion.csv"
        df = pd.read_csv(fusion_csv)
        cut_time_periods = [(None, 1647621825)]
        cut_dfs = cut_df(df, cut_time_periods)
        for i, ret_df in enumerate(cut_dfs):
            cut_df_save_path = get_save_path(fusion_csv, new_suf=f"_{i:02d}_cut.csv")
            ret_df.to_csv(cut_df_save_path, index=False)

    except Exception as e:
        logging.exception(e)
