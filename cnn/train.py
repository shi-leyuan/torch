from load_data import load_mnist_data
import matplotlib.pyplot as plt
import torch

batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 10, kernel_size=5),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=2), )
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(10, 20, kernel_size=5),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=2), )
        self.fc = torch.nn.Sequential(torch.nn.Linear(320, 50), torch.nn.ReLU(),
                                      torch.nn.Linear(50, 10))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# 创建 CNN 模型实例
model = CNN()

# 交叉熵损失
ctiterion = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ctiterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predict = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predict == target).sum().item()

        if batch_idx % 300 == 0:
            print('[%d, %5d]:loss: %.3f, acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0


# 加载 MNIST 数据集
train_loader, test_loader = load_mnist_data(batch_size=batch_size)


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d, %d]:Accuracy on test set: %1f %%' % (epoch + 1, EPOCH, 100 * acc))
    return acc


# super parameters
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

if __name__ == '__main__':
    acc_list_test = []  # 用来存储每个epoch后测试集的准确率

    # 循环训练EPOCH轮
    for epoch in range(EPOCH):
        train(epoch)  # 调用训练函数，进行一个epoch的训练

        # 每个epoch结束后测试一次
        acc_test = test(epoch)  # 测试模型在测试集上的准确率
        acc_list_test.append(acc_test)  # 保存测试准确率到列表

    # 绘制准确率曲线
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')  # x轴标签：epoch
    plt.ylabel('Accuracy On TestSet')  # y轴标签：测试集准确率
    plt.show()  # 显示图形
