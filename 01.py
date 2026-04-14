"""
对应官方教程：
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
第一部分：Datasets & Dataloaders
"""
import torch
from torch.utils.data import Dataset      # 虽然没继承，但 datasets.FashionMNIST 内部使用
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1) 下载并构建训练集
training_data = datasets.FashionMNIST(
    root="data",       # 数据集保存目录
    train=True,        # 取训练集 60 000 张
    download=True,     # 本地没有就自动下载
    transform=ToTensor()  # 把 PIL 图像转成 [0,1] 的 FloatTensor
)

# 2) 下载并构建测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,       # 取测试集 10 000 张
    download=True,
    transform=ToTensor()
)

# 3) 把数字标签映射成可读字符串
labels_map = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
}

# 4) 随机挑 9 张图可视化一下
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[idx]          # img: [1,28,28]  label: int
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")   # 去掉通道维度
plt.tight_layout()
plt.show()