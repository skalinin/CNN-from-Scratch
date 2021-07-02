import numpy as np
from cnn.model_old import (
    Conv2d, ReLU, Sigmoid, maxpool_feed, matrix2vector, fc_multiplication,
    loss_fn, fc_backpropagation, vector2matrix, maxpool_back
)


class CnnFromScratch:
    def __init__(self, train_mode, model_config, load_path):
        self.train_mode = train_mode

        self.fc_w_1 = np.array([[]])
        self.fc_b_1 = np.array([[]])
        self.fc_w_2 = np.array([[]])
        self.fc_b_2 = np.array([[]])

        self.model_config = model_config
        self.load_path = load_path

        self.conv1 = Conv2d(
            convolution=self.model_config['conv_conv_1'],
            stride=self.model_config['conv_stride_1'],
            center_w_l=self.model_config['conv_center_1'],
            in_channels=1,
            out_channels=self.model_config['conv_feature_1'],
            kernel_size=self.model_config['conv_shape_1'],
            learning_rate=self.model_config['learning_rate']
        )
        self.conv2 = Conv2d(
            convolution=self.model_config['conv_conv_2'],
            stride=self.model_config['conv_stride_2'],
            center_w_l=self.model_config['conv_center_2'],
            in_channels=self.model_config['conv_feature_1'],
            out_channels=self.model_config['conv_feature_2'],
            kernel_size=self.model_config['conv_shape_2'],
            learning_rate=self.model_config['learning_rate']
        )
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()

        self.load_weights(load_path)

    def load_weights(self, load_path):
        self.conv1.load_weights('conv_w_1', 'conv_b_1', load_path)
        self.conv2.load_weights('conv_w_2', 'conv_b_2', load_path)

    def __call__(self, image, target):
        # first conv layer
        conv_x_1 = self.conv1.feedforward(image)
        conv_y_1 = self.relu.feedforward(conv_x_1)
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
        conv_x_2 = self.conv2.feedforward(conv_y_1_mp)
        conv_y_2 = self.sigmoid1.feedforward(conv_x_2)
        # flatten feature maps to vector
        conv_y_2_vect = matrix2vector(conv_y_2)
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
            dEdconv_y_2 = vector2matrix(
                vector=dEdfc_y_0,
                matrix_shape=conv_y_2[0].shape
            )
            # second conv layer backpropagation
            dEdconv_x_2 = self.sigmoid1.backpropagation(dEdconv_y_2)
            dEdconv_y_1_mp = self.conv2.backpropagation(dEdconv_x_2)
            # maxpooling layer backpropagation
            dEdconv_y_1 = maxpool_back(
                dEdy_l_mp=dEdconv_y_1_mp,
                y_l_mp_to_y_l=conv_y_1_mp_to_conv_y_1,
                y_l_shape=conv_y_1[0].shape
            )
            # first conv layer backpropagation
            dEdconv_x_1 = self.relu.backpropagation(dEdconv_y_1)
            dEdconv_y_0 = self.conv1.backpropagation(dEdconv_x_1)
        return fc_y_2
