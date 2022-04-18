import imp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, os
import os.path as osp
# import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode, iplot


def traj_plot_by_plt(hist, fut_gt, fut_pred, file_name):
    # plt.axis('equal')
    # hist_time = np.arange(0,3,0.2)
    # fut_time = np.arange(3,8,0.5)
    fig, ax = plt.subplots(dpi=300)
    ax.set_aspect(3)
    plt.ylim(-9, 9)
    major_ticks = np.arange(-9, 9, 3.6)
    minor_ticks = np.arange(-9, 9, 1.8)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    for i in range(len(hist)):
        if i == 0:
            ax.plot(hist[i, :, 0], hist[i, :, 1], 'b', label='target_hist')
            ax.plot(hist[i, :, 0], hist[i, :, 1], '.b')
            ax.plot(hist[i, -1, 0], hist[i, -1, 1], 'or')
        else:
            ax.plot(hist[i, :, 0], hist[i, :, 1], 'k')
            ax.plot(hist[i, :, 0], hist[i, :, 1], '.k')
            ax.plot(hist[i, -1, 0], hist[i, -1, 1], 'o')

    ax.plot(fut_gt[0, :, 0], fut_gt[0, :, 1], 'g', label='fut_gt')
    ax.plot(fut_gt[0, :, 0], fut_gt[0, :, 1], '.g')
    ax.plot(fut_gt[0, -1, 0], fut_gt[0, -1, 1], 'og')

    ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], 'r', label='fut_pred')
    ax.plot(fut_pred[0, :, 0], fut_pred[0, :, 1], '.r')
    ax.plot(fut_pred[0, -1, 0], fut_pred[0, -1, 1], 'or')

    plt.title(file_name)
    plt.xlabel('Y(m)')
    plt.ylabel('X(m)')
    # plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(axis='y', linestyle='--', linewidth=2, which='major')
    ax.grid(axis='x')
    # ax.grid(linewidth=1, which='minor')
    # plt.show()
    plt.savefig(f"{file_name}")
    plt.close()


def ld_traj_plot_by_plt(hist, fut_gt, fut_pred, file_name):
    # plt.axis('equal')
    # hist_time = np.arange(0,3,0.2)
    # fut_time = np.arange(3,8,0.5)
    fig, ax = plt.subplots(dpi=300)
    # ax.set_aspect(3)
    # plt.ylim(-9, 9)
    # major_ticks = np.arange(-9, 9, 3.6)
    # minor_ticks = np.arange(-9, 9, 1.8)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    for i in range(len(hist)):
        if i == 0:
            ax.plot(hist[i, :, 1], hist[i, :, 0], 'b', label='target_hist')
            ax.plot(hist[i, :, 1], hist[i, :, 0], '.b')
            ax.plot(hist[i, -1, 1], hist[i, -1, 0], 'or')
        else:
            ax.plot(hist[i, :, 1], hist[i, :, 0], 'k')
            ax.plot(hist[i, :, 1], hist[i, :, 0], '.k')
            ax.plot(hist[i, -1, 1], hist[i, -1, 0], 'o')

    ax.plot(fut_gt[0, :, 1], fut_gt[0, :, 0], 'g', label='fut_gt')
    ax.plot(fut_gt[0, :, 1], fut_gt[0, :, 0], '.g')
    ax.plot(fut_gt[0, -1, 1], fut_gt[0, -1, 0], 'og')

    ax.plot(fut_pred[0, :, 1], fut_pred[0, :, 0], 'r', label='fut_pred')
    ax.plot(fut_pred[0, :, 1], fut_pred[0, :, 0], '.r')
    ax.plot(fut_pred[0, -1, 1], fut_pred[0, -1, 0], 'or')

    plt.title(file_name)
    # plt.xlabel('Y(m)')
    # plt.ylabel('X(m)')
    # plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(axis='y', linestyle='--', linewidth=2, which='major')
    ax.grid(axis='x')
    # ax.grid(linewidth=1, which='minor')
    # plt.show()
    plt.savefig(f"{file_name}")


# def traj_plot_by_plotly(hist, fut_gt, fut_pred):
#     traces = []
#     for i in range(len(hist)):
#         if i == 0:
#             trace = go.Scatter(x=hist[i, :, 1], y=hist[i, :, 0], mode="lines", name='ego', text='ego_hist')

#             traces.append(trace)
#         else:
#             trace = go.Scatter(x=hist[i, :, 1], y=hist[i, :, 0], mode="lines", name=f"obj{i}", text=f'obj{i}_hist')
#             traces.append(trace)

#     fut_gt_trace = go.Scatter(x=fut_gt[0, :, 1], y=fut_gt[0, :, 0], mode="lines", name=f"fut_gt", text=f'fut_gt')
#     traces.append(fut_gt_trace)
#     fut_gt_trace = go.Scatter(x=fut_pred[0, :, 1], y=fut_pred[0, :, 0], mode="lines", name=f"fut_pred", text=f'fut_pred')
#     traces.append(fut_gt_trace)

#     layout = dict(title='trajectory prediction', xaxis=dict(title="longitudance_m"))

#     fig = dict(data=traces, layout=layout)
#     fig = go.Figure(fig)
#     fig.show()


def x_magnitude(data_item):
    fut_gt = data_item.y.float()
    x_gt = fut_gt[0, :, 0]
    return max(x_gt) - min(x_gt)


def find_files(dir_path, recursive=False, suffix=".bag", prefix=''):
    all_files = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.startswith(prefix) and f.endswith(suffix):
                    # if f[-len(suffix):] == suffix and f[:len(prefix)] == prefix:
                    all_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dir_path):
            if f.startswith(prefix) and f.endswith(suffix):
                # if f[-len(suffix):] == suffix and f[:len(prefix)] == prefix:
                all_files.append(os.path.join(dir_path, f))
    if len(all_files) == 0:
        raise ValueError(f"There is no file with suffix:{suffix} and prefix:{prefix} in {dir_path}")
    all_files.sort()
    return all_files


def create_new_dir(abs_path, dir_name):
    dir_path = os.path.join(abs_path, dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def plot_traj_from_csv(csv_path, ego_id, frm_id, hist_frm=30, fut_frm=50):
    df = pd.read_csv(csv_path)
    ego_hist = df[(df['obj_id'] == ego_id) & (df['frame_id'] >= frm_id - hist_frm) & (df['frame_id'] <= frm_id)][['GlobalX', 'GlobalY']].values
    ego_fut = df[(df['obj_id'] == ego_id) & (df['frame_id'] > frm_id) & (df['frame_id'] <= frm_id + fut_frm)][['GlobalX', 'GlobalY']].values


if __name__ == '__main__':
    # hist = np.zeros((4, 31, 2))
    # hist[0, :, 0] = 0
    # hist[0, :, 1] = np.array([2 * i + 3 for i in range(31)])

    # hist[1, :, 0] = 3
    # hist[1, :, 1] = np.array([3 * i + 5 for i in range(31)])

    # hist[2, :, 0] = 6
    # hist[2, :, 1] = np.array([4 * i + 6 for i in range(31)])

    # hist[3, :, 0] = 9
    # hist[3, :, 1] = np.array([2.5 * i + 6 for i in range(31)])

    # fut_gt = np.zeros((1,10,2))
    # fut_gt[0,:,0] = 0
    # fut_gt[0,:,1] = np.array([2 * i + 3 for i in range(31)])

    # train_dir = "ngsim_data_id/ngsim_data_id_train"
    # val_dir = "ngsim_data_id/ngsim_data_id_val"
    # test_dir = "ngsim_data_id/ngsim_data_id_test"

    # for i in (train_dir, val_dir, test_dir):
    #     f = find_files(i, suffix='.pyg')
    #     print(len(f))
    # exit()
    import random
    path_dir = "/home/jiang/trajectory_pred/GNN-RNN-Based-Trajectory-Prediction-ITSC2021/stp_data_all/stp_data_all_test"
    items = find_files(path_dir, suffix='.pyg')
    cnt = 0
    for i in np.random.choice(items, 20):
        data_item = torch.load(i)
        x_mag = x_magnitude(data_item)
        if (x_mag) > 7.5:
            data_item.node_feature = data_item.node_feature.float()
            data_item.x = data_item.node_feature.float() * 0.3048
            data_item.y = data_item.y.float() * 0.3048
            print(data_item.x.size())
            print(data_item.y.size())
            noise = np.random.rand(1, 50, 2)
            y_pred = data_item.y + noise
            traj_plot_by_plt(data_item.x, data_item.y, y_pred, i.split('/')[-2:])
            # traj_plot_by_plotly(data_item.x, data_item.y, y_pred)
            print(i)
            cnt += 1
            if cnt == 20:
                break
    print(cnt)

    # data_item.node_feature = data_item.node_feature.float()
    # data_item.x = data_item.node_feature.float()
    # data_item.y = data_item.y.float()
    # print(data_item.x.size())
    # print(data_item.y.size())
    # noise = np.random.rand(1, 50, 2)
    # y_pred = data_item.y + noise
    # traj_plot_by_plt(data_item.x, data_item.y, y_pred)
    # traj_plot_by_plotly(data_item.x, data_item.y, y_pred)
