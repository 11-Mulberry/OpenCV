import os
import random

# 定义图片所在文件夹路径
image_folder = 'D:\\A8\\Reconnaissance\\P1地雷'

# 获取文件夹中所有图片的文件名列表
image_files = os.listdir(image_folder)

# 从文件名列表中随机选择20个文件名
random_files = random.sample(image_files, 2)

# 打印所选文件名，以便检查
print("随机选择的文件名列表：", random_files)

# 现在你可以修改你的代码，只对这20张图片进行检测
# 例如，你可以使用这些文件名来读取图片并进行检测
