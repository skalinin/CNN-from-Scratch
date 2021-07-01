import torch
import numpy as np
import argparse

from scripts.train_torch_model import Net


def conv_weights_converter(conv_weights):
    """Convert torch convolutional weights for the numpy model."""
    weights = []
    conv_weights = conv_weights.permute(1, 0, 2, 3)
    conv_weights = conv_weights.reshape(-1, *conv_weights.size()[2:])
    for conv_weight in conv_weights:
        weights.append(conv_weight.numpy())
    return weights


def conv_biases_converter(conv_biases):
    """Convert torch convolutional bias for the numpy model."""
    biases = []
    for conv_bias in conv_biases:
        biases.append(conv_bias.numpy().reshape(1, 1))
    return biases


def fc_weight_converter(fc_weight):
    """Convert torch fully connected weights for the numpy model."""
    return fc_weight.T.numpy()


def fc_bias_converter(fc_bias):
    """Convert torch fully connected bias for the numpy model."""
    return fc_bias.numpy().reshape(1, -1)


def main(args):
    """Make the same weights for numpy and torch models using the weights
    converter. The weights are first initialized in the torch model, and than
    converted to the numpy model.
    """
    model = Net()
    torch.save(model.state_dict(), args.torch_model_path)
    np.save(
        args.numpy_model_path,
        {
            'conv_w_1': conv_weights_converter(model.state_dict()['conv1.weight']),
            'conv_b_1': conv_biases_converter(model.state_dict()['conv1.bias']),
            'conv_w_2': conv_weights_converter(model.state_dict()['conv2.weight']),
            'conv_b_2': conv_biases_converter(model.state_dict()['conv2.bias']),
            'fc_w_1': fc_weight_converter(model.state_dict()['fc1.weight']),
            'fc_b_1': fc_bias_converter(model.state_dict()['fc1.bias']),
            'fc_w_2': fc_weight_converter(model.state_dict()['fc2.weight']),
            'fc_b_2': fc_bias_converter(model.state_dict()['fc2.bias'])
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_model_path', type=str,
                        default='/workdir/data/torch_init_weights.ckpt',
                        help='Path to save initial torch model weights')
    parser.add_argument('--numpy_model_path', type=str,
                        default='/workdir/data/numpy_init_weights.npy',
                        help='Path to save initial numpy model weights')
    args = parser.parse_args()

    main(args)
