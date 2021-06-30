import torchvision
import argparse

from cnn.transforms import PIL2numpy, Normalize, OneHot
from cnn.model import CnnFromScratch
from cnn.model_old import loss_fn


MODEL_SETTINGS = {
    'learning_rate': 0.01,
    'conv_shape_1': (2, 2),
    'conv_shape_2': (3, 3),
    'maxpool_shape_1': (2, 2),
    'conv_feature_1': 5,
    'conv_feature_2': 20,
    'conv_stride_1': 2,
    'conv_stride_2': 1,
    'maxpool_stride_1': 2,
    'fc_neurons_1': 2000,
    'conv_fn_1': 'relu',
    'conv_fn_2': 'sigmoid',
    'fc_fn_1': 'sigmoid',
    'fc_fn_2': 'softmax',
    'conv_conv_1': False,
    'conv_conv_2': False,
    'maxpool_conv_1': False,
    'conv_center_1': (0, 0),
    'conv_center_2': (1, 1),
    'maxpool_center_1': (0, 0)
}


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
    train_dataset = get_train_dataset()
    model = CnnFromScratch(
        train_mode=True,
        model_config=MODEL_SETTINGS,
        load_path=args.load_path
    )

    loss_log = []
    acc_log = []
    print_log_freq = args.print_log_freq
    for idx, (image, target) in enumerate(train_dataset):
        output = model([image], target)
        loss = loss_fn(target, output, feed=True)

        loss_log.append(loss.sum())
        acc_log.append(output.argmax() == target.argmax())
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
