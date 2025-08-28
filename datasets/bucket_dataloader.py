import math
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.transforms as T


def create_aspect_ratio_buckets(base_size=1024, multiple_of=32, max_ratio_error=0.2):
    """
    生成基于宽高比的桶列表。
    模拟 'preserve-area' 逻辑。
    
    Args:
        base_size: 基础尺寸
        multiple_of: 尺寸必须是该数的倍数
        max_ratio_error: 最大比例误差阈值
    
    Returns:
        list: 排序后的 (width, height) 桶列表
    """
    buckets = []
    target_area = base_size * base_size
    
    # 遍历可能的宽度和高度，确保是multiple_of的倍数
    for w in range(multiple_of, (base_size * 2) + 1, multiple_of):
        for h in range(multiple_of, (base_size * 2) + 1, multiple_of):
            area = w * h
            # 计算与目标面积的误差
            error = abs(area - target_area)
            buckets.append({'width': w, 'height': h, 'error': error})

    # 按误差排序找到最佳拟合的桶
    buckets.sort(key=lambda x: x['error'])
    
    # 过滤掉误差过高的桶以保持列表可管理
    max_allowed_error = buckets[int(len(buckets) * max_ratio_error)]['error']
    
    # 使用集合存储唯一的 (width, height) 元组以避免重复
    final_buckets = set()
    for bucket in buckets:
        if bucket['error'] <= max_allowed_error:
            final_buckets.add((bucket['width'], bucket['height']))

    # 按宽高比排序最终桶
    sorted_buckets = sorted(list(final_buckets), key=lambda x: x[0] / x[1])
    
    print(f"✅ 创建了 {len(sorted_buckets)} 个桶")
    return sorted_buckets


def get_all_image_files(folder_path):
    """
    递归获取文件夹及其子文件夹中的所有图片文件
    
    Args:
        folder_path: 根文件夹路径
    
    Returns:
        list: 所有图片文件的完整路径列表
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    image_files.sort()
    return image_files


class AspectRatioBucketDataset(Dataset):
    """基于宽高比桶的数据集类"""
    
    def __init__(self, image_folder_path, buckets, max_samples=None):
        """
        Args:
            image_folder_path: 图片文件夹路径（支持递归遍历子文件夹）
            buckets: 桶列表
            max_samples: 最大图像数量，用于overfitting时限制数据集大小
        """
        self.image_folder_path = image_folder_path
        self.buckets = buckets
        self.max_samples = max_samples
        
        # 递归获取所有图片文件
        print(f"正在扫描文件夹: {image_folder_path}")
        self.image_files = get_all_image_files(image_folder_path)
        
        # 如果指定了max_samples，则只使用前N张图像
        if self.max_samples is not None and self.max_samples > 0:
            self.image_files = self.image_files[:self.max_samples]
            print(f"🔒 Overfitting模式：限制使用前 {self.max_samples} 张图像")
        
        self.image_data = []  # 存储 (filepath, bucket_id)
        
        print(f"预计算图片宽高比并分配桶... 找到 {len(self.image_files)} 张图片")
        for filepath in self.image_files:
            try:
                # 获取图片尺寸而不加载完整图片
                with Image.open(filepath) as img:
                    width, height = img.size
                
                aspect_ratio = width / height
                
                # 找到最适合的桶
                best_bucket_id = min(
                    range(len(buckets)), 
                    key=lambda i: abs(aspect_ratio - (buckets[i][0] / buckets[i][1]))
                )
                
                self.image_data.append({
                    'filepath': filepath,
                    'bucket_id': best_bucket_id
                })
            except Exception as e:
                print(f"警告: 无法处理图片 {filepath}: {e}")
                continue
            
    def __len__(self):
        return len(self.image_data)
        
    def __getitem__(self, index):
        # 只返回元数据；实际的加载和调整大小将在 collate_fn 中进行
        return self.image_data[index]


class AspectRatioBucketSampler(Sampler):
    """基于宽高比桶的采样器"""
    
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            drop_last: 是否丢弃最后一个不完整的批次
            shuffle: 是否随机化顺序，True为训练模式，False为验证模式
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        # 按 bucket_id 分组索引
        self.buckets_to_indices = {}
        for idx, data in enumerate(dataset.image_data):
            bucket_id = data['bucket_id']
            if bucket_id not in self.buckets_to_indices:
                self.buckets_to_indices[bucket_id] = []
            self.buckets_to_indices[bucket_id].append(idx)
            
        # 创建所有可能的批次
        self.batches = []
        for bucket_id in self.buckets_to_indices:
            indices = self.buckets_to_indices[bucket_id]
            if self.shuffle:
                random.shuffle(indices)  # 在桶内随机打乱索引（仅训练模式）
            
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) < batch_size and self.drop_last:
                    continue
                self.batches.append(batch)
        
        # 随机打乱批次本身，使训练顺序随机（仅训练模式）
        if self.shuffle:
            random.shuffle(self.batches)
        
    def __iter__(self):
        return iter(self.batches)
        
    def __len__(self):
        return len(self.batches)


def resize_for_cropping_pil(pil_img, target_bucket_dims):
    """
    保持原始宽高比进行缩放，使缩放后的最短边等于目标桶的最短边。
    
    Args:
        pil_img: PIL图片对象
        target_bucket_dims: 目标桶尺寸 (width, height)
    
    Returns:
        PIL.Image: 调整大小后的图片
    """
    target_w, target_h = target_bucket_dims
    base_size = min(target_w, target_h)
    
    original_w, original_h = pil_img.size
    aspect_ratio = original_w / original_h

    # 使用简洁的 max() 逻辑实现 short-edge 缩放
    new_w = int(round(max(base_size * aspect_ratio, base_size)))
    new_h = int(round(max(base_size / aspect_ratio, base_size)))

    # 使用高质量的 LANCZOS 算法进行缩放
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized_img


def crop_pil(pil_img, target_dims):
    """
    中心裁剪逻辑。
    
    Args:
        pil_img: PIL图片对象
        target_dims: 目标尺寸 (width, height)
    
    Returns:
        PIL.Image: 裁剪后的图片
    """
    target_w, target_h = target_dims
    img_w, img_h = pil_img.size

    # 计算中心裁剪的坐标
    left = (img_w - target_w) // 2
    top = (img_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    cropped_img = pil_img.crop((left, top, right, bottom))
    return cropped_img


def create_cond_image(pil_image, downsample_factor=2, upsample_factor=1):
    """
    创建可调节下采样倍率的条件图像
    
    Args:
        pil_image (PIL.Image): 原始PIL图像
        downsample_factor (int): 下采样倍率，默认为2
        upsample_factor (int): 上采样倍率，默认为1
    Returns:
        torch.Tensor: 下采样的条件图像tensor (CxHxW)
    """
    # 根据下采样倍率进行下采样
    cond_width = pil_image.width // downsample_factor
    cond_height = pil_image.height // downsample_factor
    cond_pil = pil_image.resize((cond_width, cond_height), resample=Image.Resampling.BICUBIC)
    if upsample_factor > 1:
        cond_width = cond_width * upsample_factor
        cond_height = cond_height * upsample_factor
        cond_pil = cond_pil.resize((cond_width, cond_height), resample=Image.Resampling.BICUBIC)
    # 转换回tensor
    cond_tensor = T.ToTensor()(cond_pil)

    return cond_tensor


def create_collate_fn(buckets, downsample_factor=2, upsample_factor=1):
    """
    创建 collate_fn，整合 resize + crop 的完整流程。
    
    Args:
        buckets: 桶列表
        downsample_factor (int): 下采样倍率，默认为2
        upsample_factor (int): 上采样倍率，默认为1
    Returns:
        function: collate函数
    """
    def collate_fn(batch):
        # 1. 获取批次的桶信息
        bucket_id = batch[0]['bucket_id']
        target_dims = buckets[bucket_id]  # (width, height)

        processed_images = []
        processed_cond_images = []
        image_keys = []
        
        to_tensor_transform = T.Compose([
            T.ToTensor(),  # 将 PIL Image [0, 255] 转换为 Tensor [0.0, 1.0]
        ])

        # 2. 对批次中的每一张图片执行"先缩放、后裁剪"
        for item in batch:
            try:
                pil_img = Image.open(item['filepath']).convert('RGB')
                
                # 步骤 A: 保持比例缩放，为裁剪做准备
                intermediate_img = resize_for_cropping_pil(pil_img, target_dims)
                
                # 步骤 B: 进行中心裁剪，得到最终尺寸
                final_img = crop_pil(intermediate_img, target_dims)
                
                # 步骤 C: 转换为 Tensor 并归一化
                tensor_img = to_tensor_transform(final_img)
                
                # 步骤 D: 创建条件图像（使用可调节的下采样倍率）
                cond_tensor = create_cond_image(final_img, downsample_factor=downsample_factor, upsample_factor=upsample_factor)
           
                # 步骤 E: 生成图像键
                image_key = os.path.splitext(os.path.basename(item['filepath']))[0]
                
                processed_images.append(tensor_img)
                processed_cond_images.append(cond_tensor)
                image_keys.append(image_key)
                
            except Exception as e:
                print(f"警告: 处理图片时出错 {item['filepath']}: {e}")
                # 创建一个默认的黑色图片作为替代
                default_img = torch.zeros(3, target_dims[1], target_dims[0])
                default_cond_img = torch.zeros(3, target_dims[1] // downsample_factor, target_dims[0] // downsample_factor)
                default_key = f"error_{len(processed_images)}"
                
                processed_images.append(default_img)
                processed_cond_images.append(default_cond_img)
                image_keys.append(default_key)
            
        # 3. 将所有处理好的图片堆叠成一个批次
        return {
            'image': torch.stack(processed_images),  # 保持与 S3 数据格式一致
            'cond_image': torch.stack(processed_cond_images),  # 保持与 S3 数据格式一致
            '__key__': image_keys
        }
        
    print(f"✅ 已创建 Collate Function，下采样倍率: {downsample_factor}")
    return collate_fn


def create_dataloader(image_folder_path, base_size=1024, multiple_of=32, batch_size=4, 
                     num_workers=4, drop_last=False, downsample_factor=2, upsample_factor=1, shuffle=True, max_samples=None):
    """
    创建完整的 DataLoader。
    
    Args:
        image_folder_path: 图片文件夹路径
        base_size: 基础尺寸
        multiple_of: 尺寸必须是该数的倍数
        batch_size: 批次大小
        num_workers: 工作进程数
        drop_last: 是否丢弃最后一个不完整的批次
        downsample_factor (int): 下采样倍率，默认为2
        upsample_factor (int): 上采样倍率，默认为1
        shuffle (bool): 是否随机化顺序，True为训练模式，False为验证模式
        max_samples (int): 最大图像数量，用于overfitting时限制数据集大小
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    print("🚀 开始创建 DataLoader...")
    print(f"下采样倍率: {downsample_factor}, multiple_of: {multiple_of}")
    print(f"随机化模式: {'训练模式（随机）' if shuffle else '验证模式（固定顺序）'}")
    if max_samples is not None:
        print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
    
    # 1. 创建桶
    print("(1/5) 创建宽高比桶...")
    buckets = create_aspect_ratio_buckets(base_size=base_size, multiple_of=multiple_of)
    
    # 2. 创建数据集
    print("(2/5) 创建数据集...")
    dataset = AspectRatioBucketDataset(image_folder_path=image_folder_path, buckets=buckets, max_samples=max_samples)
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图片文件")
    
    # 3. 创建采样器
    print("(3/5) 创建采样器...")
    batch_sampler = AspectRatioBucketSampler(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    
    # 4. 创建 collate 函数
    print("(4/5) 创建 collate 函数...")
    collate_function = create_collate_fn(buckets, downsample_factor=downsample_factor, upsample_factor=upsample_factor)
    
    # 5. 创建 DataLoader
    print("(5/5) 创建 DataLoader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_function,
        num_workers=num_workers
    )
    
    print(f"✅ DataLoader 创建完成！数据集大小: {len(dataset)}, 批次数量: {len(batch_sampler)}")
    return dataloader


def test_dataloader(image_folder_path, num_epochs=2, max_batches=5, downsample_factor=2):
    """
    测试 DataLoader 的功能。
    
    Args:
        image_folder_path: 图片文件夹路径
        num_epochs: 测试的轮数
        max_batches: 每轮最多测试的批次数量
        downsample_factor (int): 下采样倍率，默认为2
    """
    print(f"🧪 开始测试 DataLoader...")
    print(f"图片文件夹: {image_folder_path}")
    print(f"下采样倍率: {downsample_factor}")
    
    try:
        # 创建 DataLoader
        dataloader = create_dataloader(
            image_folder_path=image_folder_path,
            base_size=1024,
            multiple_of=32,
            batch_size=2,
            num_workers=0,  # 测试时使用单进程
            drop_last=True,
            downsample_factor=downsample_factor
        )
        
        # 测试训练循环
        for epoch in range(num_epochs):
            print(f"\n--- 第 {epoch+1} 轮测试 ---")
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images_tensor = batch['image']  # 保持与 S3 数据格式一致
                cond_images_tensor = batch['cond_image']  # 条件图像
                print(f"批次 {i+1}:")
                print(f"  原始图像形状: {images_tensor.shape}")
                print(f"  条件图像形状: {cond_images_tensor.shape}")
                print(f"  数据类型: {images_tensor.dtype}")
                print(f"  数值范围: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]")
                
                # 检查是否有 NaN 或无穷大值
                if torch.isnan(images_tensor).any():
                    print("  警告: 检测到 NaN 值!")
                if torch.isinf(images_tensor).any():
                    print("  警告: 检测到无穷大值!")
                    
        print("\n✅ DataLoader 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置参数
    BASE_SIZE = 512
    MULTIPLE_OF = 64
    BATCH_SIZE = 4
    DOWNSAMPLE_FACTOR = 4  # 测试4倍下采样
    IMAGE_FOLDER = '/mnt/localssd/data/LSDIR/train/HR'  # 请修改为您的图片文件夹路径
    
    # 检查图片文件夹是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 图片文件夹不存在: {IMAGE_FOLDER}")
        print("请修改 IMAGE_FOLDER 变量为有效的图片文件夹路径")
    else:
        # 运行测试
        test_dataloader(IMAGE_FOLDER, num_epochs=2, max_batches=3, downsample_factor=DOWNSAMPLE_FACTOR)
