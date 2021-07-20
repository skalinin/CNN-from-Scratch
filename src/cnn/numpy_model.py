import numpy as np


def get_axis_indexes(kernel_axis_length, center_index):
    """Calculate the kernel indexes on one axis depending on the kernel
    center.

    Args:
        kernel_axis_length (int): The length of the single axis of the
            convolutional kernel.
        center_index (int): The index of the kernel center on one axis.
    """
    axis_indexes = []
    for i in range(-center_index, kernel_axis_length - center_index):
        axis_indexes.append(i)
    return axis_indexes


def get_axes_indexes(kernel_size, kernel_center):
    """Calculate the kernel axes indexes depending on the kernel center.

    Args:
        kernel_size (tuple of int): The size of the convolutional kernel,
            width and height, respectively.
        kernel_center (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
    """
    indexes_a = get_axis_indexes(
        kernel_axis_length=kernel_size[0],
        center_index=kernel_center[0]
    )
    indexes_b = get_axis_indexes(
        kernel_axis_length=kernel_size[1],
        center_index=kernel_center[1]
    )
    return indexes_a, indexes_b


def load_weight_from_npy(weight_name, load_path):
    weight = np.load(load_path, allow_pickle=True).item().get(weight_name)
    print(f'Weight for {weight_name} was loaded')
    return weight


class Conv2d:
    """Applies a 2D convolution over an input signal composed of several
    input planes.

    Args:
        kernel_size (tuple of int): Size of the convolving kernel.
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        stride (int): Stride of the convolution. Default is 1.
        kernel_center (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
            Default is (0, 0)
        convolution (bool): Convolution or cross-correlation will be applied.
            Default is False, which means cross-correlation.
        padding (int): Padding added to all four sides of the input. Default: 0
    """

    def __init__(
        self, kernel_size, in_channels, out_channels, stride=1,
        kernel_center=(0, 0), padding=0, convolution=False
    ):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.convolution = convolution
        self.padding = padding

        self.init_weights()

    def load_weights(self, weight_name, bias_name, load_path):
        self.conv_w = load_weight_from_npy(weight_name, load_path)
        self.conv_b = load_weight_from_npy(bias_name, load_path)

    def init_weights(self):
        self.conv_w = self.init_weight(self.kernel_size,
                                       self.in_channels*self.out_channels)
        self.conv_b = self.init_weight((1, 1), self.out_channels)

    def init_weight(self, weight_shape, weight_count):
        weight = []
        for i in range(weight_count):
            weight.append(2*np.random.random(weight_shape)-1)
        return weight

    def convolution_feed_x_l(self, y_l_minus_1, w_l, print_demo=False):
        stride = self.stride
        indexes_a, indexes_b = get_axes_indexes(w_l.shape, self.kernel_center)
        y_l_minus_1 = np.pad(y_l_minus_1, self.padding)
        h_out = int(
            (y_l_minus_1.shape[0] - (self.kernel_size[0]-1) - 1) / stride + 1
        )
        w_out = int(
            (y_l_minus_1.shape[1] - (self.kernel_size[1]-1) - 1) / stride + 1
        )
        x_l = np.zeros((h_out, w_out))
        if self.convolution:
            g = 1  # convolution
        else:
            g = -1  # cross-correlation
        for i in range(h_out):
            for j in range(w_out):
                demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]])
                result = 0
                element_exists = False
                for a in indexes_a:
                    for b in indexes_b:
                        # check that indexes of a, b did not crossed the
                        # boundaries of the y_l_minus_1
                        if (
                            i*stride - g*a >= 0
                            and j*stride - g*b >= 0
                            and i*stride - g*a < y_l_minus_1.shape[0]
                            and j*stride - g*b < y_l_minus_1.shape[1]
                        ):
                            # convert indexes of a, b to a range of positive
                            # numbers to extract data from w_l
                            result += \
                                y_l_minus_1[i*stride - g*a][j*stride - g*b] * \
                                w_l[indexes_a.index(a)][indexes_b.index(b)]
                            demo[i*stride - g*a][j*stride - g*b] = \
                                w_l[indexes_a.index(a)][indexes_b.index(b)]
                            element_exists = True
                if element_exists:
                    x_l[i][j] = result
                    # print demo matrix for tracking the convolution progress
                    if print_demo:
                        print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
        return x_l

    def convolution_back_dEdw_l(self, y_l_minus_1, dEdx_l, print_demo=False):
        stride = self.stride
        y_l_minus_1 = np.pad(y_l_minus_1, self.padding)
        w_l_shape = self.conv_w[0].shape
        indexes_a, indexes_b = get_axes_indexes(w_l_shape, self.kernel_center)
        dEdw_l = np.zeros((w_l_shape[0], w_l_shape[1]))
        if self.convolution:
            g = 1  # convolution
        else:
            g = -1  # cross-correlation
        for a in indexes_a:
            for b in indexes_b:
                demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]])
                result = 0
                for i in range(dEdx_l.shape[0]):
                    for j in range(dEdx_l.shape[1]):
                        # check that indexes of a, b did not crossed the
                        # boundaries of the y_l_minus_1
                        if (
                            i*stride - g*a >= 0
                            and j*stride - g*b >= 0
                            and i*stride - g*a < y_l_minus_1.shape[0]
                            and j*stride - g*b < y_l_minus_1.shape[1]
                        ):
                            result += \
                                y_l_minus_1[i*stride - g*a][j*stride - g*b] * \
                                dEdx_l[i][j]
                            demo[i*stride - g*a][j*stride - g*b] = \
                                dEdx_l[i][j]
                # convert indexes of a, b to a range of positive
                # numbers to extract data from w_l
                dEdw_l[indexes_a.index(a)][indexes_b.index(b)] = result
                # print demo matrix for tracking the convolution progress
                if print_demo:
                    print('a=' + str(a) + '; b=' + str(b) + '\n', demo)
        return dEdw_l

    def convolution_back_dEdy_l_minus_1(
        self, dEdx_l, w_l, y_l_minus_1_shape, print_demo=False
    ):
        indexes_a, indexes_b = get_axes_indexes(w_l.shape, self.kernel_center)
        dEdy_l_minus_1 = np.zeros((y_l_minus_1_shape[0], y_l_minus_1_shape[1]))
        if self.convolution:
            g = 1  # convolution
        else:
            g = -1  # cross-correlation
        for i in range(dEdy_l_minus_1.shape[0]):
            for j in range(dEdy_l_minus_1.shape[1]):
                result = 0
                demo = np.zeros([dEdx_l.shape[0], dEdx_l.shape[1]])
                for i_x_l in range(dEdx_l.shape[0]):
                    for j_x_l in range(dEdx_l.shape[1]):
                        a = g*i_x_l*self.stride - g*i
                        b = g*j_x_l*self.stride - g*j
                        if (
                            a in indexes_a
                            and b in indexes_b
                        ):
                            a = indexes_a.index(a)
                            b = indexes_b.index(b)
                            result += dEdx_l[i_x_l][j_x_l] * w_l[a][b]
                            demo[i_x_l][j_x_l] = w_l[a][b]
                dEdy_l_minus_1[i][j] = result
                # print demo matrix for tracking the convolution progress
                if print_demo:
                    print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
        return dEdy_l_minus_1

    def __call__(self, y_l_minus_1):
        """Feedforward of a convolutional layer.

        Args:
            y_l_minus_1 (list of numpy.ndarray): List of channels with
                (H, W)-dimension. No batch supported.
        """
        x_l = []
        for i in range(self.in_channels):
            for j in range(i*self.out_channels, (i + 1)*self.out_channels):
                # for each y_l_minus_1, the convolution is called
                # out_channels-times to create "intermediate" x_l
                x_l.append(
                    self.convolution_feed_x_l(y_l_minus_1[i], self.conv_w[j]))
        x_l_final = []
        for i in range(self.out_channels):
            x_l_final.append(0)
            for j in range(self.in_channels):
                # the "final" x_l is the sum of the "intermediate" x_l
                # received from each y_l_minus_1
                x_l_final[-1] += x_l[j*self.out_channels + i]
            # add bias to the x_l
            x_l_final[-1] += self.conv_b[len(x_l_final)-1]
        self.y_l_minus_1 = y_l_minus_1  # need for backprop
        return x_l_final

    def backprop(self, dEdx_l, learning_rate):
        """Backpropagation of a convolutional layer.

        Args:
            dEdx_l: de/dx_l matrix.
            learning_rate (float): The learning rate for training using SGD.
        """
        list_of_dEdy_l_minus_1 = []
        for i in range(self.out_channels):
            # due to the fact that one bias refers to the whole feature map,
            # dE/db_l is the sum of all the elements of dE/dx_l
            dEdb_l = dEdx_l[i].sum()
            self.conv_b[i] = self.conv_b[i] - learning_rate * dEdb_l
        for i in range(self.in_channels):
            dEdy_l_minus_1 = 0
            k = 0
            for j in range(i*self.out_channels, (i + 1)*self.out_channels):
                dEdw_l = self.convolution_back_dEdw_l(
                    y_l_minus_1=self.y_l_minus_1[i],
                    dEdx_l=dEdx_l[k],
                )
                # the backprop for dE/dy_l_minus_1 accumulates data from all
                # the corresponding feature maps
                dEdy_l_minus_1 += self.convolution_back_dEdy_l_minus_1(
                    dEdx_l=dEdx_l[k],
                    w_l=self.conv_w[j],
                    y_l_minus_1_shape=self.y_l_minus_1[i].shape,
                )
                self.conv_w[j] = self.conv_w[j] - learning_rate * dEdw_l
                k += 1
            list_of_dEdy_l_minus_1.append(dEdy_l_minus_1)
        return list_of_dEdy_l_minus_1


class Sigmoid:
    def __call__(self, x_l):
        x_l = np.array(x_l)
        y_l = 1 / (1+np.exp(-x_l))
        self.y_l = y_l  # need for backprop
        return y_l

    def backprop(self, dEdy_l):
        dy_ldx_l = self.y_l * (1 - self.y_l)
        dEdx_l = dEdy_l * dy_ldx_l
        return dEdx_l


class Softmax:
    def __call__(self, x_l):
        x_l = np.array(x_l)
        y_l = np.exp(x_l) / np.exp(x_l).sum()
        self.y_l = y_l  # need for backprop
        return y_l

    def backprop(self, dEdy_l):
        dy_ldx_l = np.zeros((self.y_l.shape[1], self.y_l.shape[1]))
        for i in range(dy_ldx_l.shape[1]):
            for j in range(dy_ldx_l.shape[1]):
                if i == j:
                    dy_ldx_l[i][i] = self.y_l[0][i]*(1 - self.y_l[0][i])
                else:
                    dy_ldx_l[i][j] = - self.y_l[0][i]*self.y_l[0][j]
        dEdx_l = np.dot(dEdy_l, dy_ldx_l)
        return dEdx_l


class ReLU:
    def __call__(self, x_l):
        x_l = np.array(x_l)
        # zero out the elements that do not pass the condition
        y_l = np.where(x_l > 0, x_l, 0)
        self.y_l = y_l  # need for backprop
        return y_l

    def backprop(self, dEdy_l):
        dy_ldx_l = np.where(self.y_l < 0, self.y_l, 1)
        # there are no negative elements in y_l after relu feedforward pass
        # so we do not neet to zero out negative elements
        # dy_ldx_l = np.where(dy_ldx_l > 0, dy_ldx_l, 0)
        dEdx_l = dEdy_l * dy_ldx_l
        return dEdx_l


class Maxpool2d:
    """Applies a 2D max pooling over an input signal composed of several
    input planes.

    Args:
        kernel_size (tuple of int): Size of the maxpooling kernel.
        stride (int): Stride of the maxpooling. Default is 1.
        kernel_center (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
            Default is (0, 0)
        convolution (book): Convolution or cross-correlation will be applied.
            Default is False, which means cross-correlation.
        padding (int): Padding added to all four sides of the input. Default: 0
    """

    def __init__(
        self, kernel_size, stride=1, kernel_center=(0, 0), padding=0,
        convolution=False
    ):
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.stride = stride
        self.convolution = convolution
        assert padding <= int(min(kernel_size) / 2), \
            "pad should be smaller than or equal to half of kernel size"
        self.padding = padding

    def maxpool(self, y_l):
        # padd by -inf to be sure these values won't be selected by maxpooling
        y_l = np.pad(y_l, self.padding, constant_values=-np.inf)
        indexes_a, indexes_b = get_axes_indexes(self.kernel_size,
                                                self.kernel_center)
        stride = self.stride
        h_out = int(
            (y_l.shape[0] - (self.kernel_size[0]-1) - 1) / stride + 1
        )
        w_out = int(
            (y_l.shape[1] - (self.kernel_size[1]-1) - 1) / stride + 1
        )
        y_l_mp = np.zeros((h_out, w_out))
        y_l_mp_to_y_l = np.zeros((h_out, w_out), dtype='<U32')
        if self.convolution:
            g = 1  # convolution
        else:
            g = -1  # cross-correlation
        for i in range(h_out):
            for j in range(w_out):
                result = -np.inf
                element_exists = False
                for a in indexes_a:
                    for b in indexes_b:
                        if (
                            i*stride - g*a >= 0
                            and j*stride - g*b >= 0
                            and i*stride - g*a < y_l.shape[0]
                            and j*stride - g*b < y_l.shape[1]
                        ):
                            if y_l[i*stride - g*a][j*stride - g*b] > result:
                                result = y_l[i*stride - g*a][j*stride - g*b]
                                # subtract self.padding from coords to make
                                # backprop correct for padded matrices
                                i_back = i*stride - g*a - self.padding
                                j_back = j*stride - g*b - self.padding
                                element_exists = True
                if element_exists:
                    y_l_mp[i][j] = result
                    y_l_mp_to_y_l[i][j] = str(i_back) + ',' + str(j_back)
        return y_l_mp, y_l_mp_to_y_l

    def __call__(self, y_l):
        """Feedforward of a maxpooling layer."""
        list_of_y_l_mp = []
        self.list_of_y_l_mp_to_y_l = []
        for i in range(len(y_l)):
            y_l_mp, y_l_mp_to_y_l = self.maxpool(y_l[i])
            list_of_y_l_mp.append(y_l_mp)
            # y_l_mp_to_y_l stores the text (coords of the selected elements
            # during feedforward) and needed for backprop: dE/dy_l_mp -> dE/dy_l
            self.list_of_y_l_mp_to_y_l.append(y_l_mp_to_y_l)
        # take an input shape to restore it on the backprop stage
        self.y_l_shape = y_l[0].shape
        return list_of_y_l_mp

    def backprop(self, dEdy_l_mp):
        """Backpropagation of a maxpooling layer."""
        list_of_dEdy_l = []
        for i in range(len(dEdy_l_mp)):
            # dEdy_l will expand as new elements are added
            dEdy_l = np.zeros(self.y_l_shape)
            for k in range(dEdy_l_mp[i].shape[0]):
                for e in range(dEdy_l_mp[i].shape[1]):
                    # each element of the dEdy_l_mp must be placed in the
                    # dEdy_l; to do this, we extract the necessary destination
                    # coordinates from the list_of_y_l_mp_to_y_l
                    coordinates = self.list_of_y_l_mp_to_y_l[i][k][e]
                    coordinate_row = int(coordinates[:coordinates.find(',')])
                    coordinate_col = int(coordinates[coordinates.find(',')+1:])
                    dEdy_l[coordinate_row][coordinate_col] = dEdy_l_mp[i][k][e]
            list_of_dEdy_l.append(dEdy_l)
        return list_of_dEdy_l


class Linear:
    """Applies a linear transformation to the incoming data.
    Fully connected layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.init_weights()

    def load_weights(self, weight_name, bias_name, load_path):
        self.fc_w = load_weight_from_npy(weight_name, load_path)
        self.fc_b = load_weight_from_npy(bias_name, load_path)

    def init_weights(self):
        self.fc_w = self.init_weight((self.in_features, self.out_features))
        self.fc_b = self.init_weight((1, self.out_features))

    def init_weight(self, weight_shape):
        weight = 2 * np.random.random(weight_shape) - 1
        return weight

    def __call__(self, y_l_minus_1):
        """Feedforward of a linear layer."""
        x_l = np.dot(y_l_minus_1, self.fc_w) + self.fc_b
        self.y_l_minus_1 = y_l_minus_1  # need for backprop
        return x_l

    def backprop(self, dEdx_l, learning_rate):
        """Backpropagation of a linear layer.

        Args:
            dEdx_l: de/dx_l matrix.
            learning_rate (float): The learning rate for training using SGD.
        """
        dEdw_l = np.dot(self.y_l_minus_1.T, dEdx_l)
        dEdb_l = dEdx_l
        dEdy_l_minus_1 = np.dot(dEdx_l, self.fc_w.T)
        self.fc_w = self.fc_w - learning_rate * dEdw_l
        self.fc_b = self.fc_b - learning_rate * dEdb_l
        return dEdy_l_minus_1


class CrossEntropyLoss:
    def __call__(self, target, predict):
        return -target * np.log(predict)

    def backprop(self, target, predict):
        return -(target/predict)


class MSELoss:
    def __call__(self, target, predict):
        return (target - predict)**2

    def backprop(self, target, predict):
        return predict - target


class Flatten:
    def matrices2vector(self, matrices):
        vector = np.array([[]])
        self.matrix_shape = matrices[0].shape
        for i in range(len(matrices)):
            reshaped_matrix = np.reshape(
                matrices[i], (1, self.matrix_shape[0]*self.matrix_shape[1]))
            vector = np.hstack((vector, reshaped_matrix))
        return vector

    def vector2matrices(self, vector):
        matrices = []
        matrix_size = self.matrix_shape[0]*self.matrix_shape[1]
        for i in range(0, vector.size, matrix_size):
            matrix = np.reshape(vector[0][i:i+matrix_size], self.matrix_shape)
            matrices.append(matrix)
        return matrices
