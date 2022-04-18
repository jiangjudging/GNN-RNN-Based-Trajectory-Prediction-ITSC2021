import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from plot_instan import save_obj_pkl

rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output.shape)

print(h0.shape)
# df = pd.read_csv("ld_data/processed_ibeo_csvs/OD_LiangDao_20220318_9988_151727_fusion_00_cut.csv")
# glb_x = df[df['obj_id'] == 0]['GlobalX'].values
# glb_y = df[df['obj_id'] == 0]['GlobalY'].values
# yaw = df[df['obj_id'] == 0]['oritentation_yaw'].values
# frm_id = df[df['obj_id'] == 0]['frame_id'].values
# # plt.plot(glb_x[450:800], glb_y[450:800])
# diff_yaw = yaw[1:] - yaw[:-1]
# mean_yaw = yaw[450:650].mean()
# print(mean_yaw)
# plt.plot(frm_id, yaw)
# # plt.plot(frm_id[1:], diff_yaw)
# plt.show()
# samples_train_file = "ngsim_samples_list/ngsim_samples_train"
# samples_val_file = "ngsim_samples_list/ngsim_samples_val"

# with open(samples_train_file, "rb") as fp:  # Unpickling
#     samples_train = pickle.load(fp)
# print(len(samples_train))

# ngsim_samples_file = "ngsim_samples_list/ngsim_samples"
# with open(ngsim_samples_file, "rb") as fp:  # Unpickling
#     ngsim_samples_list = pickle.load(fp)

# print(len(ngsim_samples_list))
# ngsim_samples_list_downsample_40 = ngsim_samples_list[::40]

# lc_samples_downsample_40 = [i for i in ngsim_samples_list_downsample_40 if i[3] == 1]
# no_lc_samples_downsample_40 = [i for i in ngsim_samples_list_downsample_40 if i[3] == 0]

# print(f"The number of lane change scenario: {len(lc_samples_downsample_40)}")
# print(f"The number of no lane change scenario: {len(no_lc_samples_downsample_40)}")
# print(f"The ratio of lc to no_lc: {len(lc_samples_downsample_40)/len(no_lc_samples_downsample_40)}")

# lc_samples_train_downsample_40, lc_samples_val_downsample_40 = train_test_split(lc_samples_downsample_40, test_size=0.1, random_state=42)
# no_lc_samples_train_downsample_40, no_lc_samples_val_downsample_40 = train_test_split(no_lc_samples_downsample_40, test_size=0.1, random_state=42)

# samples_train_downsample_40 = lc_samples_train_downsample_40 + no_lc_samples_train_downsample_40
# samples_val_downsample_40 = lc_samples_val_downsample_40 + no_lc_samples_val_downsample_40

# print(f"The number of samples_train_downsample_40: {len(samples_train_downsample_40)}")
# print(f"The number of samples_val_downsample_40: {len(samples_val_downsample_40)}")
# print(f"The ratio of lc to no_lc: {len(samples_train_downsample_40)/len(samples_val_downsample_40)}")

# with open(f"ngsim_samples_list/ngsim_samples_train_downsample_40", "wb") as fp:  #Pickling
#     pickle.dump(samples_train_downsample_40, fp)

# with open(f"ngsim_samples_list/ngsim_samples_val_downsample_40", "wb") as fp:  #Pickling
#     pickle.dump(samples_val_downsample_40, fp)

# with open("ld_data/processed_ibeo_samples_list/OD_LiangDao_20220318_9988_151727_fusion_00_cut", "rb") as fp:  # Unpickling
#     samples_val = pickle.load(fp)
# print(len(samples_val))
# print(samples_val[:10])

# print(len([i for i in samples_val if i[3] == 1]))
# print(len([i for i in samples_val if i[3] == 0]))
# del_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_40240916_04040402_G260-PDX-006-001-0402_000000-000012_LD_final__OD_MERGE_OPP/LIDAR_LJ02766_40240916_04040402_G260-PDX-006-001-0402_000000-000012_LD_final__OD_MERGE_OPP_obj_filter_interp.csv"
# del_df = pd.read_csv(del_csv)
# # start_frame = del_df[del_df['object.id'] == 244]['object list No.'].min()
# to_del_dict = {18: 8}
# start_frm_dict = {i: del_df[del_df['object.id'] == i]['object list No.'].min() for i in to_del_dict.keys()}
# print(start_frm_dict)
# # print(start_frame)
# for k, v in to_del_dict.items():
#     start_frm = start_frm_dict[k]
#     del_df = del_df.drop(del_df[(del_df['object.id'] == v) & (del_df['object list No.'] >= start_frm)].index)
# del_df.to_csv(f"{del_csv[:-4]}_del.csv", index=False)

# from preprocess_ld_data import check_dup_id_one_frm
# from plot_helper import find_files

# dataset_dirpath = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis"
# fusion_40hz_csvs = find_files(dataset_dirpath, recursive=True, suffix='fusion_40hz.csv')

# for csv in fusion_40hz_csvs:
#     # obj_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_40240916_0404044_G260-PDX-006-001-0402_000003-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_40240916_0404044_G260-PDX-006-001-0402_000003-000060_LD_final_OD_MERGE_OPP_obj.csv"
#     print(csv.split('/')[-1])
#     df = pd.read_csv(csv)
#     # print(df[df.duplicated()])
#     check_dup_id_one_frm(df)
# fusion_40hz_csvs = find_files(dataset_dirpath, recursive=True, suffix='fusion_40hz.csv')
# for csv in fusion_40hz_csvs:
#     # csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_402409140_122721_G260-PDX-006-001-0402_000000-000047_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_402409140_122721_G260-PDX-006-001-0402_000000-000047_LD_final_OD_MERGE_OPP_fusion_40hz.csv"
#     df = pd.read_csv(csv)
#     df = df.drop_duplicates()
#     df.to_csv(csv, index=False)
# print(df)
