import pandas as pd
import os
import shutil
from collections import defaultdict
import glob

root = r'C:/Users/mjynj/Desktop/process/val'
path = r'C:/Users/mjynj/Desktop/re_class/val_test_result.csv'
df = pd.read_csv(path)
df['filename'] = df['url'].apply(lambda x: x.split('/')[-1])
df['path'] = df['url'].apply(lambda x: x.split('test')[-1])
df['path'] = root + df['path']


# In[]
df1 = df[df['rv_value']>1000]

# 创建img文件夹
save_path = r'C:\Users\mjynj\Desktop\data\val\0'
os.makedirs(save_path, exist_ok=True)

# 扫描所有图片文件
all_files = {}
for folder in ['0', '1', '2', '3']:
    folder_path = os.path.join(root, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            all_files[file] = os.path.join(folder_path, file)

# 复制筛选的图片
for _, row in df1.iterrows():
    filename = row['filename']
    if filename in all_files:
        shutil.copy2(all_files[filename], os.path.join(save_path, filename))


# In[]
path = r'C:\Users\mjynj\Desktop\process\test\66'
total_path = glob.glob(os.path.join(path, '*.jpg'))
name_list = [os.path.basename(p) for p in total_path]
dff = df[df['filename'].isin(name_list)]

name_dict = dict()
for p in total_path:
    name_dict[os.path.basename(p)] = p

save_path = r'C:\Users\mjynj\Desktop\process\test'
for _, row in dff.iterrows():
    label = str(row['label'])
    filename = row['filename']
    shutil.copy2(name_dict[filename], os.path.join(save_path, label, filename))



