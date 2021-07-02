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


def get_axes_indexes(kernel_size, center_indexes):
    """Calculate the kernel axes indexes depending on the kernel center.

    Args:
        kernel_size (tuple of int): The size of the convolutional kernel,
            width and height, respectively.
        center_indexes (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
    """
    indexes_a = get_axis_indexes(
        kernel_axis_length=kernel_size[0],
        center_index=center_indexes[0]
    )
    indexes_b = get_axis_indexes(
        kernel_axis_length=kernel_size[1],
        center_index=center_indexes[1]
    )
    return indexes_a, indexes_b


def load_weight_from_npy(weight_name, load_path):
    weight = np.load(load_path, allow_pickle=True).item().get(weight_name)
    print(f'Weight for {weight_name} was loaded')
    return weight


class Conv2d:
    def __init__(
        self, stride, kernel_center, convolution, in_channels,
        out_channels, kernel_size, learning_rate
    ):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.convolution = convolution
        self.learning_rate = learning_rate

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

    def convolution_feed_x_l(self, y_l_minus_1, w_l):
        stride = self.stride
        indexes_a, indexes_b = get_axes_indexes(w_l.shape, self.kernel_center)
        # матрица выхода будет расширяться по мере добавления новых элементов
        x_l = np.zeros((1, 1))
        # в зависимости от типа операции меняется основная формула функции
        if self.convolution:
            g = 1  # операция конволюции
        else:
            g = -1  # операция корреляции
        # итерация по i и j входной матрицы y_l_minus_1 из предположения, что
        # размерность выходной матрицы x_l будет такой же
        for i in range(y_l_minus_1.shape[0]):
            for j in range(y_l_minus_1.shape[1]):
                # матрица для демонстрации конволюции
                demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]])
                result = 0
                element_exists = False
                for a in indexes_a:
                    for b in indexes_b:
                        # проверка, значения индексов не выходили за границы
                        if (
                            i*stride - g*a >= 0
                            and j*stride - g*b >= 0
                            and i*stride - g*a < y_l_minus_1.shape[0]
                            and j*stride - g*b < y_l_minus_1.shape[1]
                        ):
                            # indexes_a.index(a) перевод индексов в исходые
                            # для извлечения элементов из матрицы w_l
                            result += \
                                y_l_minus_1[i*stride - g*a][j*stride - g*b] * \
                                w_l[indexes_a.index(a)][indexes_b.index(b)]
                            demo[i*stride - g*a][j*stride - g*b] = \
                                w_l[indexes_a.index(a)][indexes_b.index(b)]
                            element_exists = True
                # запись полученных результатов только в том случае, если для
                # данных i и j были произведены вычисления
                if element_exists:
                    if i >= x_l.shape[0]:
                        # добавление строки, если не существует
                        x_l = np.vstack((x_l, np.zeros(x_l.shape[1])))
                    if j >= x_l.shape[1]:
                        # добавление столбца, если не существует
                        x_l = np.hstack((x_l, np.zeros((x_l.shape[0], 1))))
                    x_l[i][j] = result
                    # вывод матрицы demo для отслеживания хода свертки
                    # print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
        return x_l

    def convolution_back_dEdw_l(self, y_l_minus_1, w_l_shape, dEdx_l):
        stride = self.stride
        indexes_a, indexes_b = get_axes_indexes(w_l_shape, self.kernel_center)
        dEdw_l = np.zeros((w_l_shape[0], w_l_shape[1]))
        # в зависимости от типа операции меняется основная формула функции
        if self.convolution:
            g = 1  # операция конволюции
        else:
            g = -1  # операция корреляции
        # итерация по a и b ядра свертки
        for a in indexes_a:
            for b in indexes_b:
                # размерность матрицы для демонстрации конволюции равна
                # размерности y_l, так как эта матрица либо равна либо больше
                # (в случае stride>1) матрицы x_l
                demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]])
                result = 0
                for i in range(dEdx_l.shape[0]):
                    for j in range(dEdx_l.shape[1]):
                        # проверка, значения индексов не выходили за границы
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
                # indexes_a.index(a) перевод индексов в исходые
                # для извлечения элементов из матрицы w_l
                dEdw_l[indexes_a.index(a)][indexes_b.index(b)] = result
                # вывод матрицы demo для отслеживания хода свертки
                # print('a=' + str(a) + '; b=' + str(b) + '\n', demo)
        return dEdw_l

    def convolution_back_dEdy_l_minus_1(self, dEdx_l, w_l, y_l_minus_1_shape):
        indexes_a, indexes_b = get_axes_indexes(w_l.shape, self.kernel_center)
        dEdy_l_minus_1 = np.zeros((y_l_minus_1_shape[0], y_l_minus_1_shape[1]))
        # в зависимости от типа операции меняется основная формула функции
        if self.convolution:
            g = 1  # операция конволюции
        else:
            g = -1  # операция корреляции
        for i in range(dEdy_l_minus_1.shape[0]):
            for j in range(dEdy_l_minus_1.shape[1]):
                result = 0
                # матрица для демонстрации конволюции
                demo = np.zeros([dEdx_l.shape[0], dEdx_l.shape[1]])
                for i_x_l in range(dEdx_l.shape[0]):
                    for j_x_l in range(dEdx_l.shape[1]):
                        # перевод индексов в исходые для извлечения
                        # элементов из матрицы w_l + проверка на вхождение в
                        # диапазон индексов ядра свертки
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
                # вывод матрицы demo для отслеживания хода свертки
                # print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
        return dEdy_l_minus_1

    def __call__(self, y_l_minus_1):
        x_l = []
        for i in range(self.in_channels):
            for j in range(i*self.out_channels, (i + 1)*self.out_channels):
                # для каждой y_l_minus_1 функция конволюции вызывается
                # out_channels-раз для создания "промежуточных" x_l
                x_l.append(
                    self.convolution_feed_x_l(y_l_minus_1=y_l_minus_1[i],
                                              w_l=self.conv_w[j])
                )

        x_l_final = []
        for i in range(self.out_channels):
            x_l_final.append(0)
            for j in range(self.in_channels):
                # "финальный" x_l_final является суммой "промежуточных" x_l,
                # полученных с каждой y_l_minus_1
                x_l_final[-1] += x_l[j*self.out_channels + i]
            # к x_l_final прибавляем соответствующий ему bias
            x_l_final[-1] += self.conv_b[len(x_l_final)-1]

        self.y_l_minus_1 = y_l_minus_1  # need for backprop
        return x_l_final

    def backprop(self, dEdx_l):
        list_of_dEdy_l_minus_1 = []
        for i in range(self.out_channels):
            # вследствие того, что только одна b_l приходится на одну карту
            # признаков, то dEdb_l является суммой по всем элементам dEdx_l
            dEdb_l = dEdx_l[i].sum()
            # обновление b_l
            self.conv_b[i] = self.conv_b[i] - self.learning_rate * dEdb_l
        for i in range(self.in_channels):
            dEdy_l_minus_1 = 0
            k = 0
            # далее итерация по "промежуточным" картам признаков,
            # соответствующим i-тому входному каналу; сами "промежуточные"
            # карты не используются! в вычислениях присутствуют только
            # "финальные" карты; при этом количество "финальных" карт (здесь
            # dEdx_l) равно out_channels, тогда как количество w_l равно
            # out_channels*y_l_minus_1; отсюда использование дополнительного
            # "итератора" k; таким образом к dEdx_l мы обращаемся с помощью k,
            # а к w_l с помощью j
            for j in range(i*self.out_channels, (i + 1)*self.out_channels):
                dEdw_l = self.convolution_back_dEdw_l(
                    y_l_minus_1=self.y_l_minus_1[i],  # i-тый входной канал
                    w_l_shape=self.conv_w[j].shape,  # j-тый w_l
                    dEdx_l=dEdx_l[k],  # k-тый dEdx_l
                )
                # через слой y_l_minus_1 проходят суммы показателей со
                # всех карт признаков
                dEdy_l_minus_1 += self.convolution_back_dEdy_l_minus_1(
                    dEdx_l=dEdx_l[k],  # k-тый dEdx_l
                    w_l=self.conv_w[j],  # j-тый w_l
                    y_l_minus_1_shape=self.y_l_minus_1[i].shape,  # i-тый входной канал
                )
                # обновление w_l
                self.conv_w[j] = self.conv_w[j] - self.learning_rate * dEdw_l
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
                if i==j:
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
    def __init__(self, kernel_size, kernel_center, stride, convolution):
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.stride = stride
        self.convolution = convolution

    def maxpool(self, y_l):
        indexes_a, indexes_b = get_axes_indexes(self.kernel_size, self.kernel_center)
        stride = self.stride
        # выходные матрицы будут расширяться по мере добавления новых элементов
        y_l_mp = np.zeros((1,1)) # матрица y_l после операции макспулинга
        y_l_mp_to_y_l = np.zeros((1,1), dtype='<U32') # матрица для backprop через слой макспулинга (внутри матрицы будет храниться текст)
        # в зависимости от типа операции меняется основная формула функции
        if self.convolution:
            g = 1 # операция конволюции
        else:
            g = -1 # операция корреляции
        # итерация по i и j входной матрицы y_l из предположения, что размерность выходной матрицы будет такой же
        for i in range(y_l.shape[0]):
            for j in range(y_l.shape[1]):
                result = -np.inf
                element_exists = False
                for a in indexes_a:
                    for b in indexes_b:
                        # проверка, чтобы значения индексов не выходили за границы
                        if i*stride - g*a >= 0 and j*stride - g*b >= 0 \
                        and i*stride - g*a < y_l.shape[0] and j*stride - g*b < y_l.shape[1]:
                            if y_l[i*stride - g*a][j*stride - g*b] > result:
                                result = y_l[i*stride - g*a][j*stride - g*b]
                                i_back = i*stride - g*a
                                j_back = j*stride - g*b
                            element_exists = True
                # запись полученных результатов только в том случае, если для данных i и j были произведены вычисления
                if element_exists:
                    if i >= y_l_mp.shape[0]:
                        # добавление строки, если не существует
                        y_l_mp = np.vstack((y_l_mp, np.zeros(y_l_mp.shape[1])))
                        # матрица y_l_mp_to_y_l расширяется соответственно матрице y_l_mp
                        y_l_mp_to_y_l = np.vstack((y_l_mp_to_y_l, np.zeros(y_l_mp_to_y_l.shape[1])))
                    if j >= y_l_mp.shape[1]:
                        # добавление столбца, если не существует
                        y_l_mp = np.hstack((y_l_mp, np.zeros((y_l_mp.shape[0],1))))
                        y_l_mp_to_y_l = np.hstack((y_l_mp_to_y_l, np.zeros((y_l_mp_to_y_l.shape[0],1))))
                    y_l_mp[i][j] = result
                    # в матрице y_l_mp_to_y_l хранятся координаты значений,
                    # которые соответствуют выбранным в операции максипулинга ячейкам из матрицы y_l
                    y_l_mp_to_y_l[i][j] = str(i_back) + ',' + str(j_back)
        return y_l_mp, y_l_mp_to_y_l

    def __call__(self, y_l):
        list_of_y_l_mp = []
        list_of_y_l_mp_to_y_l = []
        for i in range(len(y_l)): # итерация по всем feature map в y_l
            y_l_mp, y_l_mp_to_y_l = self.maxpool(y_l[i])
            # выход функции, матрица y_l после прохождения операции макспулинга
            list_of_y_l_mp.append(y_l_mp)
            # здесь хранятся координаты, которые позволят перевести "маленькую" матрицу dE/dy_l_mp к "большой" исходной матрице dE/dy_l
            list_of_y_l_mp_to_y_l.append(y_l_mp_to_y_l)
        self.list_of_y_l_mp_to_y_l = list_of_y_l_mp_to_y_l  # need for backprop
        self.y_l_shape = y_l[0].shape  # take input shape to restore it on backprop stage
        return list_of_y_l_mp

    def backprop(self, dEdy_l_mp):
        list_of_dEdy_l = []
        for i in range(len(dEdy_l_mp)): # операция выполняется для каждой из feature map
            dEdy_l = np.zeros(self.y_l_shape) # матрица dEdy_l будет далее постепенно заполнятся значениями
            # проход по всем элементам матрицы dEdy_l_mp
            for k in range(dEdy_l_mp[i].shape[0]):
                for l in range(dEdy_l_mp[i].shape[1]):
                    # каждый элемент матрицы dEdy_l_mp необходимо поставить в матрицу dEdy_l
                    # для этого извлекаем необходимые координаты "назначения" из матрицы self.list_of_y_l_mp_to_y_l
                    coordinates = self.list_of_y_l_mp_to_y_l[i][k][l] # коордианты выглядят так: 2,4 - то есть 2-ая строка и 4-ый столбец
                    coordinate_row = int(coordinates[:coordinates.find(',')])
                    coordinate_col = int(coordinates[coordinates.find(',')+1:])
                    # запись по этим коордианатам в матрицу dEdy_l элемента из матрицы dEdy_l_mp
                    dEdy_l[coordinate_row][coordinate_col] = dEdy_l_mp[i][k][l]
            list_of_dEdy_l.append(dEdy_l) # добавляем получившуюся dEdy_l в лист с остальными feature map
        return list_of_dEdy_l


class Linear:
    def __init__(self, in_features, out_features, learning_rate):
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
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
        x_l = np.dot(y_l_minus_1, self.fc_w) + self.fc_b
        self.y_l_minus_1 = y_l_minus_1  # need for backprop
        return x_l

    def backprop(self, dEdx_l):
        # вычисление частных производных
        dEdw_l = np.dot(self.y_l_minus_1.T, dEdx_l)
        dEdb_l = dEdx_l
        dEdy_l_minus_1 = np.dot(dEdx_l, self.fc_w.T)
        # обновление матриц весов
        self.fc_w = self.fc_w - self.learning_rate * dEdw_l
        self.fc_b = self.fc_b - self.learning_rate * dEdb_l
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
