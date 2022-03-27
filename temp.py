import pandas as pd

# del_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210916_054052_G260-PDX-006-001-052_000000-000012_LD_final__OD_MERGE_OPP/LIDAR_LJ02766_20210916_054052_G260-PDX-006-001-052_000000-000012_LD_final__OD_MERGE_OPP_obj_filter_interp.csv"
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

from preprocess_ld_data import check_dup_id_one_frm
from plot_helper import find_files

dataset_dirpath = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis"
fusion_10hz_csvs = find_files(dataset_dirpath, recursive=True, suffix='fusion_10hz.csv')

for csv in fusion_10hz_csvs:
    # obj_csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210916_052044_G260-PDX-006-001-052_000003-000060_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210916_052044_G260-PDX-006-001-052_000003-000060_LD_final_OD_MERGE_OPP_obj.csv"
    print(csv.split('/')[-1])
    df = pd.read_csv(csv)
    # print(df[df.duplicated()])
    check_dup_id_one_frm(df)
# fusion_10hz_csvs = find_files(dataset_dirpath, recursive=True, suffix='fusion_10hz.csv')
# for csv in fusion_10hz_csvs:
#     # csv = "/home/jiang/trajectory_pred/ld_dataset/Dataset_for_Master_Thesis/LIDAR_LJ02766_20210915_122721_G260-PDX-006-001-052_000000-000047_LD_final_OD_MERGE_OPP/LIDAR_LJ02766_20210915_122721_G260-PDX-006-001-052_000000-000047_LD_final_OD_MERGE_OPP_fusion_10hz.csv"
#     df = pd.read_csv(csv)
#     df = df.drop_duplicates()
#     df.to_csv(csv, index=False)
# print(df)
