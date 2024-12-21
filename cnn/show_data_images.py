from load_data import load_mnist_data
import matplotlib.pyplot as plt

# 加载数据
train_loader, test_loader = load_mnist_data(batch_size=64)

# 打印训练集和测试集的大小
print(f"训练集大小: {len(train_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")

# 展示一些图像
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    img, label = train_loader.dataset[i]  # 获取单个图像和标签
    img = img.squeeze(0)  # 去除通道维度，转为 [H, W]
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title(f"Label: {label}")
    plt.xticks([])  # 不显示 x 轴坐标
    plt.yticks([])  # 不显示 y 轴坐标
plt.show()
