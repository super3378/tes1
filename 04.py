"""
对应教程：
https://docs.pytorch.org/tutorials/beginner/basics/autograd_tutorial.html
"""
import torch

# 1) 定义张量并开启梯度
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# 2) 前向计算
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 3) 查看计算图
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 4) 反向传播
loss.backward()
print(w.grad)
print(b.grad)

# 5) 禁用梯度示例
z = torch.matmul(x, w) + b
with torch.no_grad():
    z_no_grad = z + 10
print(z_no_grad.requires_grad)  # False