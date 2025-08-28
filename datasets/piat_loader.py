import torch
import functools
import piat  # 假设 piat 库已安装
import json
import os
from PIL import Image
import sys
import random
import numpy as np
from torchvision import transforms

import piat, functools

from datasets.piat_core import Dataloader
# 导入 bucket dataloader 用于本地图像训练
from datasets.bucket_dataloader import create_dataloader as create_bucket_dataloader


# def degrade_image(pil_image, mode='2x'):
#     if mode == '2x':
#         # 先4倍下采样再上采样
#         w, h = pil_image.size
#         down_w, down_h = max(1, w // 2), max(1, h // 2)
#         img = pil_image.resize((down_w, down_h), resample=Image.Resampling.BICUBIC)
#         return img
  

def attach_images(objSettings, objScratch, objOutput):
    objSettings = objSettings.copy()
    downsample_factor = objSettings.get('downsample_factor', 2)
    upsample_factor = objSettings.get('upsample_factor', 1)
    if objOutput is None:
        return 'initialized'
    if 'npyImage' in objScratch:
        pil_img = Image.fromarray(objScratch['npyImage'].copy()).convert("RGB")
        objScratch['image'] = pil_img
        # 使用可调节的下采样倍率
        cond_height = pil_img.height // downsample_factor
        cond_width = pil_img.width // downsample_factor
        objScratch['cond_image'] = transforms.Resize((cond_height, cond_width))(pil_img)
        if upsample_factor > 1:
            cond_height = cond_height * upsample_factor
            cond_width = cond_width * upsample_factor   
            objScratch['cond_image'] = transforms.Resize((cond_height, cond_width))(objScratch['cond_image'])
        
def output_data(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'image' in objScratch:
        objOutput['image'] = transforms.ToTensor()(objScratch['image'])
    if 'cond_image' in objScratch:
        objOutput['cond_image'] = transforms.ToTensor()(objScratch['cond_image'])
    if 'strImagehash' in objScratch:
        objOutput['strImagehash'] = objScratch['strImagehash']

def collate_fn(batch):
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'cond_image': torch.stack([x['cond_image'] for x in batch]),
        '__key__': [x['strImagehash'] for x in batch],
    }


def create_train_dataloader(config=None):
    """创建训练数据加载器（支持本地图像和S3数据）"""
    # 根据下采样倍率计算multiple_of
    downsample_factor = config.experiment.get("downsample_factor", 2)
    upsample_factor = config.experiment.get("upsample_factor", 1)
    multiple_of = 16 * downsample_factor

    if config is None:
        # 默认参数 - 使用S3数据
        batch_size = 5
        num_workers = 4
        num_threads = 8
        query_files = ['s3://sniklaus-clio-query/*/origin=si']
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': 1024, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 4.0', 'strCondition': 'fltGenai < 0.05'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': 1024}),
                functools.partial(piat.image_crop_smart, {'intSize': 1024}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor}),
                output_data,
            ],
            intSeed=random.randint(0, 1000000),
            collate_fn=collate_fn,
        )
    
    # 检查是否启用本地训练
    if hasattr(config, 'local_train') and config.local_train.enabled:
        print(f"使用本地图像训练，图像目录: {config.local_train.image_dir}")
        print(f"下采样倍率: {downsample_factor}, multiple_of: {multiple_of}")
        
        # 优先使用local_train中的专用参数，如果没有则回退到dataloader配置
        batch_size = getattr(config.local_train, 'batch_size', config.dataloader.train.batch_size)
        num_workers = getattr(config.local_train, 'num_workers', config.dataloader.train.num_workers)
        
        # 获取max_samples参数，用于overfitting
        max_samples = getattr(config.local_train, 'max_samples', None)
        if max_samples is not None:
            print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
        
        print(f"本地训练参数 - batch_size: {batch_size}, num_workers: {num_workers}")
        
        return create_bucket_dataloader(
            image_folder_path=config.local_train.image_dir,
            base_size=1024,  # 与 S3 数据保持一致
            multiple_of=multiple_of,   # 根据下采样倍率动态调整
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            downsample_factor=downsample_factor,
            upsample_factor=upsample_factor,
            max_samples=max_samples
        )
    else:
        # 使用S3数据
        # 从配置中读取参数
        dataloader_config = config.dataloader.train
        batch_size = dataloader_config.batch_size
        num_workers = dataloader_config.num_workers
        num_threads = dataloader_config.num_threads
        query_files = dataloader_config.query_files
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': 1024, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 4.0', 'strCondition': 'fltGenai < 0.05'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': 1024}),
                functools.partial(piat.image_crop_smart, {'intSize': 1024}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor}),
                output_data,
            ],
            intSeed=random.randint(0, 1000000),
            collate_fn=collate_fn,
        )


def create_local_train_dataloader(image_dir, batch_size=5, num_workers=4, base_size=1024, downsample_factor=2, drop_last=True, shuffle=True, max_samples=None):
    """
    创建本地训练数据加载器（使用 bucket dataloader）
    
    Args:
        image_dir (str): 图像目录路径
        batch_size (int): 批次大小，默认为5
        num_workers (int): 工作进程数，默认为4
        base_size (int): 基础尺寸，默认为1024
        downsample_factor (int): 下采样倍率，默认为2
        drop_last (bool): 是否丢弃最后一个不完整的批次，默认为True
        shuffle (bool): 是否随机化顺序，True为训练模式，False为验证模式，默认为True
        max_samples (int): 最大图像数量，用于overfitting时限制数据集大小
        
    Returns:
        DataLoader: 本地图像数据加载器
    """
    # 根据下采样倍率计算multiple_of
    multiple_of = 16 * downsample_factor
    
    return create_bucket_dataloader(
        image_folder_path=image_dir,
        base_size=base_size,
        multiple_of=multiple_of,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        downsample_factor=downsample_factor,
        shuffle=shuffle,
        max_samples=max_samples
    )


def create_val_dataloader(config=None):
    """创建验证数据加载器（支持本地图像和S3数据）"""
    # 根据下采样倍率计算multiple_of
    downsample_factor = config.experiment.get("downsample_factor", 2)
    upsample_factor = config.experiment.get("upsample_factor", 1)
    multiple_of = 16 * downsample_factor
    if config is None:
        # 默认参数 - 使用S3数据
        batch_size = 4
        num_workers = 2
        num_threads = 4
        query_files = ['s3://sniklaus-clio-query/*/origin=sf&offensiveness=0&genairemoval=date&keywordblocklist=v1']
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            intSeed=0,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': 1024, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 5.0'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': 1024}),
                functools.partial(piat.image_crop_smart, {'intSize': 1024}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor}),
                output_data,
            ],
            collate_fn=collate_fn,
        )
    
    # 检查是否启用本地评估
    if hasattr(config, 'local_eval') and config.local_eval.enabled:
        print(f"使用本地图像评估，图像目录: {config.local_eval.image_dir}")
        print(f"下采样倍率: {downsample_factor}, multiple_of: {multiple_of}")
        
        # 优先使用local_eval中的专用参数，如果没有则回退到dataloader配置
        batch_size = getattr(config.local_eval, 'batch_size', config.dataloader.val.batch_size)
        num_workers = getattr(config.local_eval, 'num_workers', config.dataloader.val.num_workers)
        
        # 获取max_samples参数，用于overfitting
        max_samples = getattr(config.local_eval, 'max_samples', None)
        if max_samples is not None:
            print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
        
        print(f"本地评估参数 - batch_size: {batch_size}, num_workers: {num_workers}")
        
        return create_bucket_dataloader(
            image_folder_path=config.local_eval.image_dir,
            base_size=1024,  # 与 S3 数据保持一致
            multiple_of=multiple_of,   # 根据下采样倍率动态调整
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,  # 验证时不丢弃最后一个批次
            downsample_factor=downsample_factor,
            upsample_factor=upsample_factor,
            shuffle=False,  # 验证集保持固定顺序
            max_samples=max_samples
        )
    else:
        # 使用S3数据
        # 从配置中读取参数
        dataloader_config = config.dataloader.val
        batch_size = dataloader_config.batch_size
        num_workers = dataloader_config.num_workers
        num_threads = dataloader_config.num_threads
        query_files = dataloader_config.query_files
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            intSeed=0,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': 1024, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 5.0'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': 1024}),
                functools.partial(piat.image_crop_smart, {'intSize': 1024}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor}),
                output_data,
            ],
            collate_fn=collate_fn,
        )


# # 为了向后兼容，保留原来的变量名
# trainDataloader = create_train_dataloader()
# valDataloader = create_val_dataloader()

if __name__ == "__main__":
    # 测试原有的验证数据加载器
    print("测试验证数据加载器:")
    for i, batch in enumerate(valDataloader):
        print(batch.keys())
        print(batch['image'].shape)
        print(batch['cond_image'].shape)
        if i > 10:
            break
    
    # 测试本地训练数据加载器（如果有测试图像目录）
    print("\n测试本地训练数据加载器:")
    try:
        test_dir = "/tmp/test_images"  # 可以修改为实际的测试目录路径
        if os.path.exists(test_dir):
            local_train_loader = create_local_train_dataloader(
                image_dir=test_dir,
                batch_size=2,
                num_workers=0,
                base_size=512,  # 测试时使用较小的尺寸
                downsample_factor=4
            )
            
            for i, batch in enumerate(local_train_loader):
                print(f"批次 {i}:")
                print(f"  图像形状: {batch['image'].shape}")  # 保持与 S3 数据格式一致
                print(f"  条件图像形状: {batch['cond_image'].shape}")  # 保持与 S3 数据格式一致
                print(f"  图像key: {batch['__key__']}")
                if i >= 2:
                    break
            print("本地训练数据加载器测试完成!")
        else:
            print(f"测试目录 {test_dir} 不存在，跳过本地训练数据加载器测试")
    except Exception as e:
        print(f"本地训练数据加载器测试失败: {e}")