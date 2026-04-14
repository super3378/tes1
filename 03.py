"""
对应教程：
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

# 1) 指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 2) 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 3) 数据加载（与官网保持一致）
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 4) 实例化并打印模型
model = NeuralNetwork().to(device)
print(model)

# 5) 快速前向测试
X, y = next(iter(train_dataloader))
X = X.to(device)
logits = model(X)
print(logits.shape)  # 期望 [64, 10]