import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision

from cnn.transforms import PIL2numpy, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 2, 2)
        self.conv2 = nn.Conv2d(5, 20, 3, 1, padding=1)
        self.fc1 = nn.Linear(980, 2000)
        self.fc2 = nn.Linear(2000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x  # log_softmax already in F.cross_entropy


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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    return train_loader


def main(args):
    train_loader = get_train_loader()
    model = Net()
    if args.load_path:
        states = torch.load(args.load_path)
        model.load_state_dict(states)
        print('Model weights were loaded')
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_log = []
    acc_log = []
    print_log_freq = args.print_log_freq
    for idx, (image, target) in enumerate(train_loader):
        image = image.unsqueeze(0)  # Add channel to make input 4D
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        loss_log.append(loss)
        acc_log.append(output.argmax().item() == target.item())
        if idx % print_log_freq == 0:
            loss_avg = sum(loss_log)/len(loss_log)
            acc_avg = sum(acc_log)/len(acc_log)
            loss_log = []
            acc_log = []
            print(f'Step {idx}, Loss: {loss_avg:.4f}, '
                  f'Accyracy: {acc_avg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_log_freq', type=int, default=30,
                        help='Frequency of printing of training logs')
    parser.add_argument('--load_path', type=str, default='',
                        help='Path to model weights to start training with')
    args = parser.parse_args()

    main(args)
