from matplotlib import pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import pickle
import pandas as pd


def plot_glb_frm(hists, fut_pred, fut_gt, save_path=None, xlim=None, ylim=None):
    """
    画一帧的图

    Parameters
    ----------
    hists : np.array(n*30*2)
        n表示车辆的数目，其中第0个必须是目标车辆的数据，后面的是周围车辆的轨迹
        30表示过去30帧
        2表示x,y坐标
    fut_pred : 50*2
        目标车辆的预测未来轨迹
        50表示未来50帧
        2表示x,y坐标
    fut_gt : 50*2  
        目标车辆的实际未来轨迹
        50表示未来50帧
        2表示x,y坐标
    save_path : string, optional
        照片保存路径, by default None
    xlim : tuple, optional
        x坐标轴的范围, by default None
    ylim : tuple, optional
        y坐标轴的范围, by default None
    """
    fig, ax = plt.subplots(dpi=300)
    # fig.set_tight_layout(True)

    # ax.set_aspect(3)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    for i, veh_hist in enumerate(hists):
        if i == 0:
            ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'b', label='target_hist')
            ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.b')
            ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'or')
        else:
            ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'k')
            ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.k')
            ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'o')

    # ax.plot(fut_gt[:, 0], fut_gt[:, 1], 'g', label='fut_gt')
    # ax.plot(fut_gt[:, 0], fut_gt[:, 1], '.g')
    # ax.plot(fut_gt[-1, 0], fut_gt[-1, 1], 'og')

    ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], 'r', label='fut_pred')
    ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], '.r')
    ax.plot(fut_pred[0, -1, 0], fut_pred[0, -1, 1], 'or')

    # file_name = save_path.split('/')[-1][3:-4]
    # plt.title(file_name)
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    # plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    # plt.savefig(save_path, facecolor='white', edgecolor='none')
    # plt.close()


def make_video(hists_dict, fut_pred_dict, fut_gt_dict, duration, save_path, fps=10):
    """
    生成视频

    Parameters
    ----------
    hists_dict : dict
        历史轨迹的字典，key为帧id，value为历史轨迹（n*30*2）
    fut_pred_dict : dict
        预测的未来轨迹字典，key为帧id，value为预测的未来轨迹（50*2）
    fut_gt_dict : dict
        实际的未来轨迹字典，key为帧id，value为实际的未来轨迹（50*2）
    duration : float/int
        视频时长
    save_path : string 
        保存路径
    fps : int, optional
        每秒帧数, by default 10
    """

    def make_frame(t):
        frm = int(10 * t)
        hists = hists_dict[frm]
        fut_pred = fut_pred_dict[frm]
        fut_gt = fut_gt_dict[frm]
        fig, ax = plt.subplots(dpi=300)
        plt.xlim(-100, 250)
        plt.ylim(-8, 4)
        for i, veh_hist in enumerate(hists):
            if i == 0:
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'b', label='target_hist')
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.b')
                ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'or')
            else:
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'k')
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.k')
                ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'o')

        ax.plot(fut_gt[:, 0], fut_gt[:, 1], 'g', label='fut_gt')
        ax.plot(fut_gt[:, 0], fut_gt[:, 1], '.g')
        ax.plot(fut_gt[-1, 0], fut_gt[-1, 1], 'og')

        ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], 'r', label='fut_pred')
        ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], '.r')
        ax.plot(fut_pred[0, -1, 0], fut_pred[0, -1, 1], 'or')

        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.close()
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(save_path, fps=fps)


def make_full_pred_video(hists_dict, fut_pred_dict, fut_gt_dict, duration, save_path, fps=10):
    """
    生成视频

    Parameters
    ----------
    hists_dict : dict
        历史轨迹的字典，key为帧id，value为历史轨迹（n*30*2）
    fut_pred_dict : dict
        预测的未来轨迹字典，key为帧id，value为预测的未来轨迹（50*2）
    fut_gt_dict : dict
        实际的未来轨迹字典，key为帧id，value为实际的未来轨迹（50*2）
    duration : float/int
        视频时长
    save_path : string 
        保存路径
    fps : int, optional
        每秒帧数, by default 10
    """

    def make_frame(t):
        frm = int(10 * t) + 450
        hists = hists_dict[frm]
        fut_preds = fut_pred_dict[frm]
        fut_gts = fut_gt_dict[frm]
        fig, ax = plt.subplots(dpi=300)
        # plt.xlim(-150, 75)
        # plt.ylim(-4, 4)
        for i, veh_hist in enumerate(hists):
            # print(veh_hist)
            if i == 0:
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'b', label='ego_hist')
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.b')
                ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'or')
            else:
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], 'k')
                ax.plot(veh_hist[:, 0], veh_hist[:, 1], '.k')
                ax.plot(veh_hist[-1, 0], veh_hist[-1, 1], 'o')

        for j, fut_pred in enumerate(fut_preds):
            ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], 'r', label='fut_pred')
            ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], '.r')
            ax.plot(fut_pred[0, -1, 0], fut_pred[0, -1, 1], 'or')
            break

        for k, fut_gt in enumerate(fut_gts):
            ax.plot(fut_gt[:, 0], fut_gt[:, 1], 'g')
            ax.plot(fut_gt[:, 0], fut_gt[:, 1], '.g')
            ax.plot(fut_gt[-1, 0], fut_gt[-1, 1], 'og')
            break

        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        # return
        plt.close()
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(save_path, fps=fps)


def get_glb_hist_poc_id(df, tgt_id, frm_id, hist_len=30):
    '''
    得到指定id，指定帧的历史轨迹
    '''
    tgt_hist_pos = df[(df['obj_id'] == tgt_id) & (df['frame_id'] >= frm_id - hist_len) & (df['frame_id'] <= frm_id)][['GlobalX', 'GlobalY']].values
    return tgt_hist_pos


def get_glb_fut_poc_id(df, tgt_id, frm_id, fut_len=50):
    '''
    得到指定id，指定帧的未来
    '''
    tgt_hist_pos = df[(df['obj_id'] == tgt_id) & (df['frame_id'] > frm_id) & (df['frame_id'] <= frm_id + fut_len)][['GlobalX', 'GlobalY']].values
    return tgt_hist_pos[4::5, :]


def get_ego_yaw(df, frm_id):
    '''
    得到指定frm_id的主车的yaw角
    '''
    yaw = df[(df['obj_id'] == 0) & (df['frame_id'] == frm_id)]['oritentation_yaw'].values[0]
    return yaw


def get_ego_pos(df, frm_id):
    '''
    得到指定frm_id的主车的全局位置
    '''
    ref_pos = df[(df['obj_id'] == 0) & (df['frame_id'] == frm_id)][['GlobalX', 'GlobalY']].values[0]
    return ref_pos


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


def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with open(f"imgs/ld_eval_imgs/fut_pred_dict/ibeo_ds2_pred2.pkl", "rb") as fp:  #Pickling
        ibeo_ds1_pred_dict = pickle.load(fp)
    frm_ids = sorted(list(ibeo_ds1_pred_dict[2].keys()))
    frm_ids = [i for i in frm_ids if 450 <= i <= 650]
    print(frm_ids)
    df = pd.read_csv("ld_data/processed_ibeo_csvs/OD_LiangDao_20220318_9988_151727_fusion_00_cut.csv")

    # frm_ids = set(frm_ids[:300])
    # print(frm_ids[:300])

    ibeo_ds1_hist_dict = {}
    ibeo_ds1_fut_gt_dict = {}
    for ds_id, frms in ibeo_ds1_pred_dict.items():
        for frm_id in frm_ids:
            frm = frms[frm_id]
            tgt_ids = list(frm.keys())
            ibeo_ds1_fut_gt_dict[frm_id] = {}
            ibeo_ds1_hist_dict[frm_id] = {}
            for tgt_id in tgt_ids:
                glb_hist_pos = get_glb_hist_poc_id(df, tgt_id, frm_id)
                glb_fut_pos = get_glb_fut_poc_id(df, tgt_id, frm_id)
                ibeo_ds1_hist_dict[frm_id][tgt_id] = glb_hist_pos
                ibeo_ds1_fut_gt_dict[frm_id][tgt_id] = glb_fut_pos

    ibeo_ds1_rotate_dict = {'rotate_glb_hist': {}, "rotate_fut_pred": {}, "rotate_fut_gt": {}}
    for ds_id, frms in ibeo_ds1_pred_dict.items():
        for frm_id in frm_ids:
            frm = frms[frm_id]
            tgt_ids = sorted(list(frm.keys()))

            yaw = get_ego_yaw(df, frm_id)
            # yaw = 2.488
            ref_pos = get_ego_pos(df, frm_id)
            frm_rotate_hists = []
            frm_fut_pred = []
            frm_fut_gt = []
            for tgt_id in tgt_ids:
                glb_hist_pos = ibeo_ds1_hist_dict[frm_id][tgt_id] - ref_pos
                # rotate_glb_pos = rotate_points(glb_hist_pos, -yaw)
                # frm_rotate_hists.append(rotate_glb_pos)
                frm_rotate_hists.append(glb_hist_pos)

                if tgt_id != 0:
                    fut_pred = glb_hist_pos[-1, :] + ibeo_ds1_pred_dict[2][frm_id][tgt_id]
                    frm_fut_pred.append(fut_pred)
                    fut_gt = ibeo_ds1_fut_gt_dict[frm_id][tgt_id] - ref_pos
                    frm_fut_gt.append(fut_gt)
                    # rotate_fut_gt = rotate_points(fut_gt, -yaw)
                    # frm_fut_gt.append(rotate_fut_gt)
                    # rotate_fut_pred = rotate_points(fut_pred, -yaw)
                    # frm_fut_pred.append(rotate_fut_pred)
            ibeo_ds1_rotate_dict['rotate_glb_hist'][frm_id] = frm_rotate_hists
            ibeo_ds1_rotate_dict['rotate_fut_pred'][frm_id] = frm_fut_pred
            ibeo_ds1_rotate_dict['rotate_fut_gt'][frm_id] = frm_fut_gt

    save_obj_pkl(ibeo_ds1_rotate_dict, "imgs/ld_eval_imgs/fut_pred_dict/ibeo_ds2_no_rotate_gt")
    with open(f"imgs/ld_eval_imgs/fut_pred_dict/ibeo_ds2_no_rotate_gt.pkl", "rb") as fp:  #Pickling
        ibeo_ds1_rotate_dict = pickle.load(fp)

    hists_dict = ibeo_ds1_rotate_dict['rotate_glb_hist']
    fut_pred_dict = ibeo_ds1_rotate_dict['rotate_fut_pred']
    fut_gt_dict = ibeo_ds1_rotate_dict['rotate_fut_gt']
    make_full_pred_video(hists_dict, fut_pred_dict, fut_gt_dict, 20, 'imgs/ld_eval_imgs/ibeo_ds1/ibeo_ds2_full_pred_no_rotate.mp4')

    # for tgt_id, fut_pred in tgts.items():
    #     print(tgt_id)

    # hists_dict, fut_pred_dict, fut_gt_dict = {}, {}, {}
    # for t in range(100):
    #     hists = ds7_id36_dict[1140 + t]["rotate_glb_pos"]
    #     fut_gt = ds7_id36_dict[1140 + t]["rotate_fur_gt"]
    #     fut_pred = ds7_id36_dict[1140 + t]["rotate_fut_pred"]

    #     hists_dict[t] = hists
    #     fut_pred_dict[t] = fut_pred
    #     fut_gt_dict[t] = fut_gt
    #     # break

    # duration = 10
    # save_path = "/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/imgs/ld_eval_imgs/ds7_id36/ds7_id36_1140_1240_3.mp4"
    # make_video(hists_dict, fut_pred_dict, fut_gt_dict, duration, save_path)
