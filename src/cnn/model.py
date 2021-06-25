import numpy as np
from cnn.model_old import (
    convolution_feed, maxpool_feed, matrix2vector_tf, fc_multiplication,
    loss_fn, fc_backpropagation, vector2matrix_tf, convolution_backpropagation,
    maxpool_back
)


class CnnFromScratch:
    def __init__(self, train_mode, model_config, load_path):
        self.train_mode = train_mode
        self.conv_w_1 = []
        self.conv_b_1 = []
        self.conv_w_2 = []
        self.conv_b_2 = []
        self.fc_w_1 = np.array([[]])
        self.fc_b_1 = np.array([[]])
        self.fc_w_2 = np.array([[]])
        self.fc_b_2 = np.array([[]])

        self.model_config = model_config
        self.load_path = load_path

    def save_weights(self, save_path):
        np.save(
            save_path,
            {
                'conv_w_1': self.conv_w_1,
                'conv_b_1': self.conv_b_1,
                'conv_w_2': self.conv_w_2,
                'conv_b_2': self.conv_b_2,
                'fc_w_1': self.fc_w_1,
                'fc_b_1': self.fc_b_1,
                'fc_w_2': self.fc_w_2,
                'fc_b_2': self.fc_b_2
            }
        )

    def __call__(self, image, target):
        # first conv layer
        conv_y_1, self.conv_w_1, self.conv_b_1 = convolution_feed(
            y_l_minus_1=image,
            w_l=self.conv_w_1,
            w_l_name='conv_w_1',
            w_shape_l=self.model_config['conv_shape_1'],
            b_l=self.conv_b_1,
            b_l_name='conv_b_1',
            feature_maps=self.model_config['conv_feature_1'],
            act_fn=self.model_config['conv_fn_1'],
            dir_npy=self.load_path,
            conv_params={
                'convolution': self.model_config['conv_conv_1'],
                'stride': self.model_config['conv_stride_1'],
                'center_w_l': self.model_config['conv_center_1']
            }
        )
        # maxpooling layer
        conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = maxpool_feed(
            y_l=conv_y_1,
            conv_params={
                'window_shape': self.model_config['maxpool_shape_1'],
                'convolution': self.model_config['maxpool_conv_1'],
                'stride': self.model_config['maxpool_stride_1'],
                'center_window': self.model_config['maxpool_center_1']
            }
        )
        # second conv layer
        conv_y_2, self.conv_w_2, self.conv_b_2 = convolution_feed(
            y_l_minus_1=conv_y_1_mp,
            w_l=self.conv_w_2,
            w_l_name='conv_w_2',
            w_shape_l=self.model_config['conv_shape_2'],
            b_l=self.conv_b_2,
            b_l_name='conv_b_2',
            feature_maps=self.model_config['conv_feature_2'],
            act_fn=self.model_config['conv_fn_2'],
            dir_npy=self.load_path,
            conv_params={
                'convolution': self.model_config['conv_conv_2'],
                'stride': self.model_config['conv_stride_2'],
                'center_w_l': self.model_config['conv_center_2']
            }
        )
        # flatten feature maps to vector
        conv_y_2_vect = matrix2vector_tf(conv_y_2)
        # first fully connected layer
        fc_y_1, self.fc_w_1, self.fc_b_1 = fc_multiplication(
            y_l_minus_1=conv_y_2_vect,
            w_l=self.fc_w_1,
            w_l_name='fc_w_1',
            b_l=self.fc_b_1,
            b_l_name='fc_b_1',
            neurons=self.model_config['fc_neurons_1'],
            act_fn=self.model_config['fc_fn_1'],
            dir_npy=self.load_path
        )
        # second fully connected layer
        fc_y_2, self.fc_w_2, self.fc_b_2 = fc_multiplication(
            y_l_minus_1=fc_y_1,
            w_l=self.fc_w_2,
            w_l_name='fc_w_2',
            b_l=self.fc_b_2,
            b_l_name='fc_b_2',
            neurons=len(target),
            act_fn=self.model_config['fc_fn_2'],
            dir_npy=self.load_path
        )
        if self.train_mode:
            # loss function backpropagation
            dEdfc_y_2 = loss_fn(target, fc_y_2, feed=False)
            # second fully connected layer backpropagation
            dEdfc_y_1, self.fc_w_2, self.fc_b_2 = fc_backpropagation(
                y_l_minus_1=fc_y_1,
                dEdy_l=dEdfc_y_2,
                y_l=fc_y_2,
                w_l=self.fc_w_2,
                b_l=self.fc_b_2,
                act_fn=self.model_config['fc_fn_2'],
                alpha=self.model_config['learning_rate']
            )
            # first fully connected layer backpropagation
            dEdfc_y_0, self.fc_w_1, self.fc_b_1 = fc_backpropagation(
                y_l_minus_1=conv_y_2_vect,
                dEdy_l=dEdfc_y_1,
                y_l=fc_y_1,
                w_l=self.fc_w_1,
                b_l=self.fc_b_1,
                act_fn=self.model_config['fc_fn_1'],
                alpha=self.model_config['learning_rate']
            )
            # convert vector to feature maps
            dEdconv_y_2 = vector2matrix_tf(
                vector=dEdfc_y_0,
                matrix_shape=conv_y_2[0].shape
            )
            # second conv layer backpropagation
            dEdconv_y_1_mp, self.conv_w_2, self.conv_b_2 = convolution_backpropagation(
                y_l_minus_1=conv_y_1_mp,
                y_l=conv_y_2,
                w_l=self.conv_w_2,
                b_l=self.conv_b_2,
                dEdy_l=dEdconv_y_2,
                feature_maps=self.model_config['conv_feature_2'],
                act_fn=self.model_config['conv_fn_2'],
                alpha=self.model_config['learning_rate'],
                conv_params={
                    'convolution': self.model_config['conv_conv_2'],
                    'stride': self.model_config['conv_stride_2'],
                    'center_w_l': self.model_config['conv_center_2']
                }
            )
            # maxpooling layer backpropagation
            dEdconv_y_1 = maxpool_back(
                dEdy_l_mp=dEdconv_y_1_mp,
                y_l_mp_to_y_l=conv_y_1_mp_to_conv_y_1,
                y_l_shape=conv_y_1[0].shape
            )
            # first conv layer backpropagation
            dEdconv_y_0, self.conv_w_1, self.conv_b_1 = convolution_backpropagation(
                y_l_minus_1=image,
                y_l=conv_y_1,
                w_l=self.conv_w_1,
                b_l=self.conv_b_1,
                dEdy_l=dEdconv_y_1,
                feature_maps=self.model_config['conv_feature_1'],
                act_fn=self.model_config['conv_fn_1'],
                alpha=self.model_config['learning_rate'],
                conv_params={
                    'convolution': self.model_config['conv_conv_1'],
                    'stride': self.model_config['conv_stride_1'],
                    'center_w_l': self.model_config['conv_center_1']
                }
            )
        return fc_y_2


def get_axis_indexes(kernel_axis_length, center_index):
    """Calculate the kernel indexes on a certain axis depending on the kernel
    center.

    Args:
        kernel_axis_length (int): The length of the single axis of the
            convolutional kernel.
        center_index (int): The index of the kernel center on a certain axis.
    """
    axis_indexes = []
    for i in range(-center_index, kernel_axis_length - center_index):
        axis_indexes.append(i)
    return axis_indexes


def get_axes_indexes(kernel_size, center_indexes):
    """Calculate the kernel axes indexes depending on the kernel center.

    Args:
        kernel_size (tuple of int): The size of the convolutional kernel. The
            first index should be on the x-axis, and the second on the y-axis.
        center_indexes (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
    """
    indexes_x = get_axis_indexes(
        kernel_axis_length=kernel_size[0],
        center_index=center_indexes[0]
    )
    indexes_y = get_axis_indexes(
        kernel_axis_length=kernel_size[1],
        center_index=center_indexes[1]
    )
    return indexes_x, indexes_y
