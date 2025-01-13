import os
import wave

def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            # 检查文件是否可以被成功打开
            wf.getparams()  # 获取文件参数，如果文件不符合 WAV 格式会抛出异常
        return True
    except wave.Error:
        return False

def count_non_standard_wav_files(directory):
    non_standard_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                if not is_valid_wav(file_path):
                    non_standard_count += 1
    return non_standard_count

# 输入目录路径
directory = 'E:\Multimodal-analysis-of-infant-crying\data'

non_standard_count = count_non_standard_wav_files(directory)

if non_standard_count > 0:
    print(f"共有 {non_standard_count} 个文件不是标准的WAV格式。")
else:
    print("所有文件都是标准的WAV格式。")
