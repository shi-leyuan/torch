from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_mnist_data(batch_size=64):
    """
    加载 MNIST 数据集并进行标准化处理

    :param batch_size: 每个批次的大小
    :return: train_loader, test_loader 数据加载器
    """
    # 数据预处理：转换为张量
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载 MNIST 训练集
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

    # 使用 DataLoader 加载训练集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # 初始化用于计算均值和标准差的变量
    mean = 0.0
    std = 0.0
    total_images = 0

    # 计算训练集的均值和标准差
    for images, _ in train_loader:
        batch_samples = images.size(0)  # 获取当前批次的样本数
        images = images.view(batch_samples, images.size(1), -1)  # 展平每个图像的像素
        mean += images.mean(2).sum(0)  # 计算每个通道的均值
        std += images.std(2).sum(0)  # 计算每个通道的标准差
        total_images += batch_samples
    # 归一化均值和标准差
    mean = mean / total_images
    std = std / total_images
    # 定义数据预处理（转换为张量，并标准化）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    # 加载标准化后的训练集和测试集
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    # 使用 DataLoader 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
