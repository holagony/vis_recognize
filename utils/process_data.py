import os
import glob
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_date_and_filename(url):
    try:
        parts = url.split('/')
        if len(parts) >= 2:
            date_folder = parts[-2]
            filename = parts[-1]
            return date_folder, filename
    except:
        pass


def process_data(file_path, csv_path, i):
    file_names = [os.path.basename(f) for f in glob.glob(os.path.join(file_path, '*.jpg'))]
    
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, encoding='gbk')
        
    df[['date_folder', 'filename']] = df['img_url'].apply(lambda x: pd.Series(extract_date_and_filename(x)))
    df_unique = df.drop_duplicates(subset=['filename'], keep='first')
    out = df_unique[df_unique['filename'].isin(file_names)]
    out = out[['camera_name', 'camera_num', 'camera_id', 'lon', 'lat', 'observe_time', 'visibility', 'level', 'filename']]
    
    if i in [0, 1]:
        out['company'] = '成宜'
        
    elif i == 2:
        level_mapping = {1: 4, 2: 3, 3: 2, 4: 1}
        out['level'] = out['level'].map(level_mapping)
        out['company'] = '荣乌'
    
    elif i in [3,4]:
        out['company'] = '荣乌'
        
    elif i == 5:
        out['company'] = '石太'    
        
    return out


file_list = [r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\成宜2023',
             r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\成宜2024',
             r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌2023',
             r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌2024-2025',
             r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌2024-2025',
             r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\石太2025']

csv_list = [r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\成宜低能见度(20230102-20231206).csv',
            r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\成宜低能见度(20240510-20240612).csv',
            r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌低能见度(20230120-20230220).csv',
            r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌低能见度(20240510-20241229).csv',
            r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\荣乌低能见度(20250101-20250916).csv',
            r'C:\Users\mjynj\Desktop\img_data\data\能见度\低能见度样本和标签\石太低能见度(20250724-20250916).csv']

# 低能见度
out_list = []
for i in range(6):
    csv_path = csv_list[i]
    file_path = file_list[i]
    out = process_data(file_path, csv_path, i)
    out_list.append(out)

out_df = pd.concat(out_list, axis=0)
out_df = out_df[out_df['level']!=5]
out_df['observe_time'] = pd.to_datetime(out_df['observe_time'])
out_df['level'].value_counts()

# 高能见度
file_path = r'C:\Users\mjynj\Desktop\img_data\data\能见度\level5\level5'
csv_path = r'C:/Users/mjynj/Desktop/img_data/data/能见度/level5_visibility_samples_120k.csv'
file_names = [os.path.basename(f) for f in glob.glob(os.path.join(file_path, '*.jpg'))]
df = pd.read_csv(csv_path)
df[['date_folder', 'filename']] = df['img_url'].apply(lambda x: pd.Series(extract_date_and_filename(x)))
df_unique = df.drop_duplicates(subset=['filename'], keep='first')
out = df_unique[df_unique['filename'].isin(file_names)]
out['camera_num'] = np.nan
out = out[['camera_name', 'camera_num', 'camera_id', 'lon', 'lat', 'observe_time', 'visibility', 'level', 'filename', 'company']]
out['observe_time'] = pd.to_datetime(out['observe_time'])
out.loc[out['company']=='蜀道集团', 'company'] = '成宜'
out['level'].value_counts()

# 高低能见度样本合并
vis_df = pd.concat([out_df,out],axis=0)
vis_df.sort_values(by='observe_time', inplace=True)
vis_df['year'] = vis_df['observe_time'].dt.year

# In[]
# 划分数据集
clip =  vis_df[~((vis_df['year']<2025) | (vis_df['company']=='石太'))]
train_df = vis_df[(vis_df['year']<2025) | (vis_df['company']=='石太')]
val_df = clip[clip['observe_time']<'2025-08']
test_df = clip[clip['observe_time']>='2025-08']

# In[]
train_df.to_csv(r'C:/Users/mjynj/Desktop/train.csv',index=False,encoding='utf-8')
val_df.to_csv(r'C:/Users/mjynj/Desktop/val.csv',index=False,encoding='utf-8')
test_df.to_csv(r'C:/Users/mjynj/Desktop/test.csv',index=False,encoding='utf-8')

# --------------------------------------------------------------------
# 分类存放
def read_csv_files():
    """
    从./data/label文件夹读取train.csv、val.csv、test.csv为dataframe
    """
    csv_files = ['train.csv', 'val.csv', 'test.csv']
    dataframes = {}
    
    for csv_file in csv_files:
        csv_path = f'./data/label/{csv_file}'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dataframes[csv_file.replace('.csv', '')] = df
            print(f"成功读取 {csv_file}，包含 {len(df)} 条记录")
        else:
            print(f"警告：文件 {csv_path} 不存在")
    
    return dataframes

def extract_and_organize_images(dataframes, source_folder='D:/data/level5'):
    """
    根据dataframe的filename列从源文件夹提取图片，
    并根据level列的数值创建文件夹结构进行分类
    """
    for csv_name, df in dataframes.items():
        print(f"\n处理 {csv_name} 数据集...")
        
        # 检查必要的列是否存在
        if 'filename' not in df.columns or 'level' not in df.columns:
            print(f"错误：{csv_name} 缺少必要的列 (filename 或 level)")
            continue
        
        # 创建目标根目录
        target_root = f'./data/{csv_name}'
        os.makedirs(target_root, exist_ok=True)
        
        # 统计信息
        processed_count = 0
        missing_count = 0
        
        # 使用tqdm显示进度条
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {csv_name}", unit="张"):
            filename = row['filename']
            level = int(row['level'])  # 确保level为整数
            
            # 源图片路径
            source_path = os.path.join(source_folder, filename)
            
            # 检查源图片是否存在
            if not os.path.exists(source_path):
                tqdm.write(f"警告：图片 {filename} 在源文件夹中不存在")
                missing_count += 1
                continue
            
            # 创建level对应的目标文件夹
            level_folder = os.path.join(target_root, str(level))
            os.makedirs(level_folder, exist_ok=True)
            
            # 目标图片路径
            target_path = os.path.join(level_folder, filename)
            
            try:
                # 复制图片
                shutil.copy2(source_path, target_path)
                processed_count += 1
                    
            except Exception as e:
                tqdm.write(f"复制图片 {filename} 时出错：{e}")
        
        print(f"{csv_name} 处理完成：")
        print(f"  - 成功处理：{processed_count} 张图片")
        print(f"  - 缺失图片：{missing_count} 张")
        print(f"  - 目标文件夹：{target_root}")

def get_folder_structure(dataframes):
    """
    显示将要创建的文件夹结构
    """
    print("\n将要创建的文件夹结构：")
    for csv_name, df in dataframes.items():
        if 'level' in df.columns:
            unique_levels = sorted([int(level) for level in df['level'].unique()])
            print(f"\n{csv_name}:")
            for level in unique_levels:
                count = len(df[df['level'] == level])
                print(f"  ./data/{csv_name}/{level}/ ({count} 张图片)")

def main():
    """
    主函数：执行整个图片分类整理流程
    """
    print("开始图片分类整理程序...")
    print("=" * 50)
    
    # 步骤1：读取CSV文件
    print("步骤1：读取CSV文件")
    dataframes = read_csv_files()
    
    if not dataframes:
        print("错误：没有成功读取任何CSV文件")
        return
    
    # 显示文件夹结构预览
    get_folder_structure(dataframes)
    
    # 确认是否继续
    response = input("\n是否继续执行图片复制操作？(y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 步骤2：提取和整理图片
    print("\n步骤2：提取和整理图片")
    extract_and_organize_images(dataframes)
    
    print("\n" + "=" * 50)
    print("图片分类整理完成！")

if __name__ == "__main__":
    main()







