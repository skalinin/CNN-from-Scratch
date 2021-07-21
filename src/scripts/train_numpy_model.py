import torchvision
import argparse
import time

from cnn.transforms import PIL2numpy, Normalize, OneHot
from cnn.numpy_model import (
    Conv2d, ReLU, Sigmoid, Softmax, Maxpool2d, Flatten, Linear,
    CrossEntropyLoss
)


class CnnFromScratch:
    def __init__(self):
        self.conv1 = Conv2d(
            stride=1,
            in_channels=1,
            out_channels=5,
            kernel_size=(2, 2),
        )
        self.conv2 = Conv2d(
            stride=2,
            in_channels=5,
            out_channels=20,
            kernel_size=(3, 3),
            padding=1,
        )
        self.max_pool = Maxpool2d(
            kernel_size=(2, 2),
            stride=2,
            padding=1
        )
        self.fc1 = Linear(980, 2000)
        self.fc2 = Linear(2000, 10)
        self.flatten = Flatten()
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()
        self.sigmoid2 = Sigmoid()
        self.softmax = Softmax()

    def load_weights(self, load_path):
        self.conv1.load_weights('conv_w_1', 'conv_b_1', load_path)
        self.conv2.load_weights('conv_w_2', 'conv_b_2', load_path)
        self.fc1.load_weights('fc_w_1', 'fc_b_1', load_path)
        self.fc2.load_weights('fc_w_2', 'fc_b_2', load_path)

    def __call__(self, image, target):
        x = self.conv1(image)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.sigmoid1(x)
        x = self.flatten.matrices2vector(x)
        x = self.fc1(x)
        x = self.sigmoid2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def backprop(self, x, learning_rate=0.01):
        x = self.softmax.backprop(x)
        x = self.fc2.backprop(x, learning_rate)
        x = self.sigmoid2.backprop(x)
        x = self.fc1.backprop(x, learning_rate)
        x = self.flatten.vector2matrices(x)
        x = self.sigmoid1.backprop(x)
        x = self.conv2.backprop(x, learning_rate)
        x = self.max_pool.backprop(x)
        x = self.relu.backprop(x)
        x = self.conv1.backprop(x, learning_rate)


def get_train_dataset():
    transforms = torchvision.transforms.Compose([
        PIL2numpy(),
        Normalize(),
    ])
    target_transform = torchvision.transforms.Compose([
        OneHot()
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='/workdir/data',
        train=True,
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    return train_dataset


def main(args):
    learning_rate = 0.01
    train_dataset = get_train_dataset()
    model = CnnFromScratch()
    if args.load_path:
        model.load_weights(args.load_path)
    criterion = CrossEntropyLoss()

    loss_log = []
    acc_log = []
    print_log_freq = args.print_log_freq
    start_time = time.time()
    for idx, (image, target) in enumerate(train_dataset):
        predict = model([image], target)
        loss = criterion(target, predict)

        x = criterion.backprop(target, predict)
        model.backprop(x, learning_rate)

        loss_log.append(loss.sum())
        acc_log.append(predict.argmax() == target.argmax())
        if idx % print_log_freq == 0:
            loss_avg = sum(loss_log)/len(loss_log)
            acc_avg = sum(acc_log)/len(acc_log)
            loss_log = []
            acc_log = []
            loop_time = time.time() - start_time
            start_time = time.time()
            print(f'Step {idx}, Loss: {loss_avg:.4f}, '
                  f'Accyracy: {acc_avg:.4f}, loop time, sec: {loop_time:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_log_freq', type=int, default=30,
                        help='Frequency of printing of training logs')
    parser.add_argument('--load_path', type=str, default='',
                        help='Path to model weights to start training with')
    args = parser.parse_args()

    main(args)
