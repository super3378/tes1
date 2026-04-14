"""
对应教程：
https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader

# 1) 定义变换组合：ToTensor + Normalize
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# 2) 重新加载数据集并应用变换
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,          # 与官网一致
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# 3) 构造 DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 4) 简单测试
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")