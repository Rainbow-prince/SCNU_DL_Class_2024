import os
import argparse
import numpy as np
# from d2l.mxnet import accuracy
# from sympy.stats.sampling.sample_scipy import scipy
from torch.utils import data
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard.summary import image
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from model_cnn_TODO import Net
from PIL import Image

import matplotlib.pyplot as plt


def train(args, model, device, train_loader, optimizer, epoch) -> None:
    # TODO: explain the function of train()
    """
    用于训练模型。遍历训练数据集，计算损失，并通过反向传播更新模型的权重
    @param args:    超参数。包括 '--type’, '--batch-size', '--test-batch-size', '--epochs', '--lr'
    @param model:   要训练的神经网络模型
    @param device:  CPU或GPU..
    @param train_loader:    从训练数据集中批量加载数据
    @param optimizer:       优化器，比如SGD或Adam
    @param epoch:   训练周期
    @return:        None
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Lr:{:.4f} Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item()))


# phase indicate validating or testing
# note: during training, you can only validate
def test(model, device, test_loader, phase='validate'):
    # TODO: explain the function of eval()
    """
    评估模型性能。计算整个数据集上的平均损失和准确率
    @param model:       要训练的神经网络模型
    @param device:      CPU或GPU..
    @param test_loader: 从训练数据集中批量加载测试用的数据
    @param phase:  训练时选择'validate'，测试时为'Test'
    @return: test_accuracy 对于具体的某一个epoch的正确率
    """

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            # TODO: explain the function of argmax
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            """  
            模型会分别输出结果logits，表示对于某个输入数据，其为各个种类的概率大小。
            argmax会找到最大值的索引，即模型认为最可能的类别
            """
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\n {}: Average loss: {:.4f}, Accuracy: {}/{} \t ({:.0f}%)\n'.format(
        phase,
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))

    return test_accuracy

# read source https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST
class mnist_dataset(data.Dataset):
    def __init__(self, data: np.array, label: np.array, transform: transforms):
        super(mnist_dataset, self).__init__()
        self.data = data
        self.label = label.astype(np.int64)
        self.transform = transform

        # sanity check
        assert self.data.shape[0] == self.label.shape[0]

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, index: int):
        """
        从data中取出一个array，转换成灰度图像之后，并进行transfrom（转成tensor并归一化）
        在把tensor变形  [1,784]->(1, 28, 28)

        @param index:   要获取的样本的索引
        @return:        (1, 28, 28)灰度图像以及标签
        """
        image = self.data[index, :]  # 从数据集中取出索引为index的图像数据，此时图像还是一个np array
        label = self.label[index, 0]  # 对应的标签，一个整数

        # TODO: explain each line of reading in a gray image
        # use `Image` to convert to image and apply normalization
        image = Image.fromarray(image, mode="L")  # mode="L"：灰度模式
        image = self.transform(image)  # 在其它地方已经定义的transform操作：包括把图像转化成tensor，以及归一化
        # image [1,784] convert to a 3-D matrix of size (1,28,28)
        # the first channel 1 says its gray image with channel 1
        # if you process RGB image, that will be 3-channel instead 1
        image = image.squeeze().view(1, 28, 28)  # 变形
        return image, label


def load_mnist_data(args):
    # download mnist numpy data files from
    # https://www.kaggle.com/datasets/sivasankaru/mnist-npy-file-dataset?resource=download

    # 直接点击运行时路径如下：
    train_labels = np.load('data/train_labels.npy')
    train_images = np.load('data/train_images.npy')
    test_labels = np.load('data/test_labels.npy')
    test_images = np.load('data/test_images.npy')

    # (10000, 1) (60000, 1)
    print(test_labels.shape, train_labels.shape)
    # (10000, 784) (60000, 784)
    print(test_images.shape, train_images.shape)

    # TODO: explain transforms of images
    # transforms.Compose 用于将多个图像变换操作组合成一个序列
    transform = transforms.Compose([
        transforms.ToTensor(),  # 把图像转换成tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 规范化，减去均值、除以方差
    ])

    # Q1_step1
    # 在进行训练和测试之前先进行可视化
    # image_1 = train_images[0,:]  # 取出第一张图像
    # image_1 =  np.reshape(image_1, (28, 28))  # 变形
    # image_1 = Image.fromarray(image_1, mode="L")  # 转化成PIL类型
    # image_1.save("vis.png")  # 保存图像 完成可视化

    train_data = mnist_dataset(train_images, train_labels, transform)
    test_data = mnist_dataset(test_images, test_labels, transform)

    train_size = int(0.8 * len(train_images))
    validate_size = len(train_images) - train_size
    print('Split training data into %d for training and %d for validation' % (train_size, validate_size))

    train_dataset, validate_dataset = torch.utils.data.random_split(train_data, [train_size, validate_size])
    print(len(train_dataset), len(validate_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(len(train_loader), len(validate_loader), len(test_loader))
    return train_loader, validate_loader, test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--type', type=str, default='SGD', choices=['SGD', 'ADAM'],
                        help='use SGD or ADAM')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # if you have a GPU, turn this on
    use_cuda = args.cuda and torch.cuda.is_available()

    # if you have a MAC M1/2 chip, turn this on
    use_mps = False  # args.mps or torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('The device is', device)

    seed = 47
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load mnist data
    train_loader, validate_loader, test_loader = load_mnist_data(args)

    model = Net().to(device)

    if args.type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise "This model type is not implemented !!"

    scheduler = StepLR(optimizer, step_size=8, gamma=args.gamma)

    # training phase, use validation set to check performance
    # TODO: record the validation accuracy in each epoch
    # use Matplot to draw the accuracy changes over each epoch
    # compare SGD, ADAM convergence and explain which and why is better

    # 存储每个epoch的验证准确率
    accuracies = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuray_i = test(model, device, validate_loader, phase='Validate')
        accuracies.append(accuray_i)

        scheduler.step()

    # testing phase, use test set to evaluate the final performance
    print('========= Final Testing =============')
    test(model, device, test_loader, phase='Test')

    # save model
    if args.save_model:
        if not os.path.isdir('output'):
            os.mkdir('output')
        torch.save(model.state_dict(), "output/mnist_%s.pt" % (args.type,))

    # 对准确率的变化进行可视化
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
