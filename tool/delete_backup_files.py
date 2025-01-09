import os

def delete_backup_files(directory):
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以 '-副本.wav' 结尾
            if file.endswith(' - 副本.wav'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # 删除文件
                    deleted_count += 1
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件时出错 {file_path}: {e}")
    return deleted_count

# 输入你想检查的目录路径
directory = 'E:\Multimodal-analysis-of-infant-crying\data\hungry'  # 修改为实际的目录路径

deleted_count = delete_backup_files(directory)

if deleted_count > 0:
    print(f"共删除 {deleted_count} 个 ' - 副本.wav' 文件。")
else:
    print("没有找到 '-副本.wav' 文件，未进行删除。")
