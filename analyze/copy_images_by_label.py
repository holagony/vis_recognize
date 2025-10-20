#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据CSV文件中的label和filename，将相应的图片复制到result文件夹下
先在val文件夹中查找，如果没有找到再在test文件夹中查找
在result中保留val/test文件夹结构
"""

import pandas as pd
import os
import shutil
from pathlib import Path


def copy_images_by_label():
    """
    根据CSV文件复制图片到result文件夹，先在val中查找，再在test中查找
    """
    # 文件路径配置
    csv_file = r'c:\Users\mjynj\Desktop\traffic\vis_recognize\img_data\reclass\val_test_reclass.csv'
    val_base_dir = r'c:\Users\mjynj\Desktop\traffic\vis_recognize\img_data\data\val'
    test_base_dir = r'c:\Users\mjynj\Desktop\traffic\vis_recognize\img_data\data\test'
    result_base_dir = r'c:\Users\mjynj\Desktop\traffic\vis_recognize\result'

    df = pd.read_csv(csv_file)
    os.makedirs(result_base_dir, exist_ok=True)

    # 统计变量
    copied_count = 0
    not_found_count = 0
    error_count = 0
    val_found_count = 0
    test_found_count = 0

    # 处理每条记录
    for idx, row in df.iterrows():
        filename = row['filename']
        pred_value = row['pred']
        label_value = row['label']

        # 先在val文件夹中查找
        val_source_file = os.path.join(val_base_dir, str(label_value), filename)
        test_source_file = os.path.join(test_base_dir, str(label_value), filename)

        source_file = None
        source_type = None

        if os.path.exists(val_source_file):
            source_file = val_source_file
            source_type = 'val'
            val_found_count += 1
        elif os.path.exists(test_source_file):
            source_file = test_source_file
            source_type = 'test'
            test_found_count += 1
        else:
            print(f"文件未找到: {filename} (pred: {pred_value})")
            not_found_count += 1
            continue

        # 创建对应的目标目录结构 result/val/pred 或 result/test/pred
        target_dir = os.path.join(result_base_dir, source_type, str(pred_value))
        os.makedirs(target_dir, exist_ok=True)

        # 构建目标文件路径
        target_file = os.path.join(target_dir, filename)

        try:
            # 复制文件
            shutil.copy2(source_file, target_file)
            copied_count += 1
            if copied_count % 100 == 0:
                print(f"已复制 {copied_count} 个文件...")

        except Exception as e:
            print(f"复制文件时出错 {source_file}: {str(e)}")
            error_count += 1

    # 输出统计结果
    print(f"\n复制完成!")
    print(f"成功复制: {copied_count} 个文件")
    print(f"  - 从val文件夹找到: {val_found_count} 个文件")
    print(f"  - 从test文件夹找到: {test_found_count} 个文件")
    print(f"未找到: {not_found_count} 个文件")
    print(f"错误: {error_count} 个文件")
    print(f"总处理: {copied_count + not_found_count + error_count} 条记录")

    # 输出各文件夹的文件数量
    print(f"\nresult文件夹中的文件分布:")
    for source_type in ['val', 'test']:
        source_dir = os.path.join(result_base_dir, source_type)
        if os.path.exists(source_dir):
            print(f"\n{source_type}文件夹:")
            for pred in df['pred'].unique():
                pred_dir = os.path.join(source_dir, str(pred))
                if os.path.exists(pred_dir):
                    file_count = len([f for f in os.listdir(pred_dir) if f.endswith('.jpg')])
                    if file_count > 0:
                        print(f"  Pred {pred}: {file_count} 个文件")

if __name__ == "__main__":
    copy_images_by_label()
