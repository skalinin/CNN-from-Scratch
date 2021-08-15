import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
import time

from cnn.transforms import PIL2numpy, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.conv2 = nn.Conv2d(2, 5, 2, 2, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(320, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x  # log_softmax is in CrossEntropyLoss


def get_train_loader():
    transforms = torchvision.transforms.Compose([
        PIL2numpy(),
        Normalize(),
        ToTensor()
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='/workdir/data',
        train=True,
        download=True,
        transform=transforms
    )
    test_dataset = torchvision.datasets.MNIST(
        root='/workdir/data',
        train=False,
        download=True,
        transform=transforms
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_loader, test_loader


def train_loop(dataset, model, criterion, optimizer, print_log_freq):
    loss_log = []
    acc_log = []
    start_time = time.time()
    model.train()
    for idx, (image, target) in enumerate(dataset):
        image = image.unsqueeze(0)  # Add channel to make input 4D
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        acc_log.append(pred.argmax().item() == target.item())
        if idx % print_log_freq == 0:
            loss_avg = sum(loss_log[-print_log_freq:])/print_log_freq
            acc_avg = sum(acc_log[-print_log_freq:])/print_log_freq
            loop_time = time.time() - start_time
            start_time = time.time()
            print(f'Train step {idx}, Loss: {loss_avg:.5f}, '
                  f'Acc: {acc_avg:.4f}, time: {loop_time:.1f}')


def val_loop(dataset, model, criterion):
    loss_log = []
    acc_log = []
    start_time = time.time()
    model.eval()
    for idx, (image, target) in enumerate(dataset):
        image = image.unsqueeze(0)  # Add channel to make input 4D
        with torch.no_grad():
            pred = model(image)
            loss = criterion(pred, target)
        loss_log.append(loss.item())
        acc_log.append(pred.argmax().item() == target.item())

    loss_avg = sum(loss_log)/len(loss_log)
    acc_avg = sum(acc_log)/len(acc_log)
    loop_time = time.time() - start_time
    print(f'Val step, Loss: {loss_avg:.5f}, '
          f'Acc: {acc_avg:.4f}, time: {loop_time:.1f}')


def main(args):
    train_loader, test_loader = get_train_loader()
    model = Net()
    if args.load_path:
        states = torch.load(args.load_path)
        model.load_state_dict(states)
        print('Torch model weights were loaded')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        train_loop(train_loader, model, criterion,
                   optimizer, args.print_log_freq)
        val_loop(test_loader, model, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_log_freq', type=int, default=1000,
                        help='Frequency of printing of training logs')
    parser.add_argument('--load_path', type=str, default='',
                        help='Path to model weights to start training with')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Total number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    args = parser.parse_args()

    main(args)
