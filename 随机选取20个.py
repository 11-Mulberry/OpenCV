import os
import random
import shutil

# 定义图片所在文件夹路径
image_folder = 'D:\\A8\\Reconnaissance\\P1地雷'
target_folder = 'D:\\A8\\Reconnaissance\\新建文件夹'

# 获取文件夹中所有图片的文件名列表
image_files = os.listdir(image_folder)

# 从文件名列表中随机选择20个文件名
random_files = random.sample(image_files, 5)

# 遍历随机选择的文件名列表，复制文件到目标文件夹
for file_name in random_files:
    source_path = os.path.join(image_folder, file_name)
    target_path = os.path.join(target_folder, file_name)
    shutil.copyfile(source_path, target_path)

# 打印所选文件名，以便检查
print("随机选择的文件名列表：", random_files)
