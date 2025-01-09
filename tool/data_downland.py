# import os
# from kaggle.api.kaggle_api_extended import KaggleApi
#
# # 初始化 Kaggle API
# api = KaggleApi()
# api.authenticate()
#
# # 设置下载数据集的目标路径
# dataset_name = "chrisfilo/urbansound8k"  # UrbanSound8K 数据集
# destination = "/dataset"  # 数据集保存的路径
#
# # 下载并解压数据集
# api.dataset_download_files(dataset_name, path=destination, unzip=True)
#
# print(f"数据集已下载到 {destination}")

import kagglehub

# Download latest version
path = kagglehub.dataset_download("chrisfilo/urbansound8k")

print("Path to dataset files:", path)