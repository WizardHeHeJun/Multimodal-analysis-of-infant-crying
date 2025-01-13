import os
from tool.is_valid_wav import is_valid_wav


def delete_non_standard_wav_files(directory):
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                if not is_valid_wav(file_path):
                    try:
                        os.remove(file_path)  # 删除不符合标准的WAV文件
                        deleted_count += 1
                        print(f"已删除不标准的WAV文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件时出错 {file_path}: {e}")
    return deleted_count

# 输入目录路径
directory = 'E:\Multimodal-analysis-of-infant-crying\data'

deleted_count = delete_non_standard_wav_files(directory)

if deleted_count > 0:
    print(f"共删除 {deleted_count} 个不标准的WAV文件。")
else:
    print("没有发现不标准的WAV文件，未进行删除。")
