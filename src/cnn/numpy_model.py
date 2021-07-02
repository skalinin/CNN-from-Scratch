from cnn.model_old import (
    Conv2d, ReLU, Sigmoid, Softmax, Maxpool2d, Flatten, Linear
)


class CnnFromScratch:
    def __init__(self, config):
        self.conv1 = Conv2d(
            convolution=config['conv_conv_1'],
            stride=config['conv_stride_1'],
            kernel_center=config['conv_center_1'],
            in_channels=1,
            out_channels=config['conv_feature_1'],
            kernel_size=config['conv_shape_1'],
            learning_rate=config['learning_rate']
        )
        self.conv2 = Conv2d(
            convolution=config['conv_conv_2'],
            stride=config['conv_stride_2'],
            kernel_center=config['conv_center_2'],
            in_channels=config['conv_feature_1'],
            out_channels=config['conv_feature_2'],
            kernel_size=config['conv_shape_2'],
            learning_rate=config['learning_rate']
        )
        self.max_pool = Maxpool2d(
            kernel_size=config['maxpool_shape_1'],
            kernel_center=config['maxpool_center_1'],
            stride=config['maxpool_stride_1'],
            convolution=config['maxpool_conv_1']
        )
        self.fc1 = Linear(
            in_features=config['fc_1_in_features'],
            out_features=config['fc_1_out_features'],
            learning_rate=config['learning_rate']
        )
        self.fc2 = Linear(
            in_features=config['fc_2_in_features'],
            out_features=config['fc_2_out_features'],
            learning_rate=config['learning_rate']
        )
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

    def backprop(self, x):
        x = self.softmax.backprop(x)
        x = self.fc2.backprop(x)
        x = self.sigmoid2.backprop(x)
        x = self.fc1.backprop(x)
        x = self.flatten.vector2matrices(x)
        x = self.sigmoid1.backprop(x)
        x = self.conv2.backprop(x)
        x = self.max_pool.backprop(x)
        x = self.relu.backprop(x)
        x = self.conv1.backprop(x)
