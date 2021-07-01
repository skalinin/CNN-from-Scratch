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


def convolution_feed_x_l(y_l_minus_1, w_l, conv_params):
	indexes_a, indexes_b = get_axes_indexes(w_l.shape, conv_params['center_w_l'])
	stride = conv_params['stride']
	# матрица выхода будет расширяться по мере добавления новых элементов
	x_l = np.zeros((1,1))
	# в зависимости от типа операции меняется основная формула функции
	if conv_params['convolution']:
		g = 1 # операция конволюции
	else:
		g = -1 # операция корреляции
	# итерация по i и j входной матрицы y_l_minus_1 из предположения, что размерность выходной матрицы x_l будет такой же
	for i in range(y_l_minus_1.shape[0]):
		for j in range(y_l_minus_1.shape[1]):
			demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]]) # матрица для демонстрации конволюции
			result = 0
			element_exists = False
			for a in indexes_a:
				for b in indexes_b:
					# проверка, чтобы значения индексов не выходили за границы
					if i*stride - g*a >= 0 and j*stride - g*b >= 0 \
					and i*stride - g*a < y_l_minus_1.shape[0] and j*stride - g*b < y_l_minus_1.shape[1]:
						result += y_l_minus_1[i*stride - g*a][j*stride - g*b] * w_l[indexes_a.index(a)][indexes_b.index(b)] # перевод индексов в "нормальные" для извлечения элементов из матрицы w_l
						demo[i*stride - g*a][j*stride - g*b] = w_l[indexes_a.index(a)][indexes_b.index(b)]
						element_exists = True
			# запись полученных результатов только в том случае, если для данных i и j были произведены вычисления
			if element_exists:
				if i >= x_l.shape[0]:
					# добавление строки, если не существует
					x_l = np.vstack((x_l, np.zeros(x_l.shape[1])))
				if j >= x_l.shape[1]:
					# добавление столбца, если не существует
					x_l = np.hstack((x_l, np.zeros((x_l.shape[0],1))))
				x_l[i][j] = result
				# вывод матрицы demo для отслеживания хода свертки
				# print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
	return x_l

def maxpool(y_l, conv_params):
	indexes_a, indexes_b = get_axes_indexes(conv_params['window_shape'], conv_params['center_window'])
	stride = conv_params['stride']
	# выходные матрицы будут расширяться по мере добавления новых элементов
	y_l_mp = np.zeros((1,1)) # матрица y_l после операции макспулинга
	y_l_mp_to_y_l = np.zeros((1,1), dtype='<U32') # матрица для backprop через слой макспулинга (внутри матрицы будет храниться текст)
	# в зависимости от типа операции меняется основная формула функции
	if conv_params['convolution']:
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

def maxpool_feed(y_l, conv_params):
	list_of_y_l_mp = []
	list_of_y_l_mp_to_y_l = []
	for i in range(len(y_l)): # итерация по всем feature map в y_l
		y_l_mp, y_l_mp_to_y_l = maxpool(y_l[i], conv_params)
		# выход функции, матрица y_l после прохождения операции макспулинга
		list_of_y_l_mp.append(y_l_mp)
		# здесь хранятся координаты, которые позволят перевести "маленькую" матрицу dE/dy_l_mp к "большой" исходной матрице dE/dy_l
		list_of_y_l_mp_to_y_l.append(y_l_mp_to_y_l)
	return list_of_y_l_mp, list_of_y_l_mp_to_y_l

def maxpool_back(dEdy_l_mp, y_l_mp_to_y_l, y_l_shape):
	list_of_dEdy_l = []
	for i in range(len(dEdy_l_mp)): # операция выполняется для каждой из feature map
		dEdy_l = np.zeros(y_l_shape) # матрица dEdy_l будет далее постепенно заполнятся значениями
		# проход по всем элементам матрицы dEdy_l_mp
		for k in range(dEdy_l_mp[i].shape[0]):
			for l in range(dEdy_l_mp[i].shape[1]):
				# каждый элемент матрицы dEdy_l_mp необходимо поставить в матрицу dEdy_l
				# для этого извлекаем необходимые координаты "назначения" из матрицы y_l_mp_to_y_l
				coordinates = y_l_mp_to_y_l[i][k][l] # коордианты выглядят так: 2,4 - то есть 2-ая строка и 4-ый столбец
				coordinate_row = int(coordinates[:coordinates.find(',')])
				coordinate_col = int(coordinates[coordinates.find(',')+1:])
				# запись по этим коордианатам в матрицу dEdy_l элемента из матрицы dEdy_l_mp
				dEdy_l[coordinate_row][coordinate_col] = dEdy_l_mp[i][k][l]
		list_of_dEdy_l.append(dEdy_l) # добавляем получившуюся dEdy_l в лист с остальными feature map
	return list_of_dEdy_l

def convolution_back_dEdw_l(y_l_minus_1, w_l_shape, dEdx_l, conv_params):
	indexes_a, indexes_b = get_axes_indexes(w_l_shape, conv_params['center_w_l'])
	stride = conv_params['stride']
	dEdw_l = np.zeros((w_l_shape[0], w_l_shape[1]))
	# в зависимости от типа операции меняется основная формула функции
	if conv_params['convolution']:
		g = 1 # операция конволюции
	else:
		g = -1 # операция корреляции
	# итерация по a и b ядра свертки
	for a in indexes_a:
		for b in indexes_b:
			# размерность матрицы для демонстрации конволюции равноа размерности y_l, так как эта матрица либо равна либо больше (в случае stride>1) матрицы x_l
			demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]])
			result = 0
			for i in range(dEdx_l.shape[0]):
				for j in range(dEdx_l.shape[1]):
					# проверка, чтобы значения индексов не выходили за границы
					if i*stride - g*a >= 0 and j*stride - g*b >= 0 \
					and i*stride - g*a < y_l_minus_1.shape[0] and j*stride - g*b < y_l_minus_1.shape[1]:
						result += y_l_minus_1[i*stride - g*a][j*stride - g*b] * dEdx_l[i][j]
						demo[i*stride - g*a][j*stride - g*b] = dEdx_l[i][j]
			dEdw_l[indexes_a.index(a)][indexes_b.index(b)] = result # перевод индексов в "нормальные" для извлечения элементов из матрицы w_l
			# вывод матрицы demo для отслеживания хода свертки
			# print('a=' + str(a) + '; b=' + str(b) + '\n', demo)
	return dEdw_l

def convolution_back_dEdy_l_minus_1(dEdx_l, w_l, y_l_minus_1_shape, conv_params):
	indexes_a, indexes_b = get_axes_indexes(w_l.shape, conv_params['center_w_l'])
	stride = conv_params['stride']
	dEdy_l_minus_1 = np.zeros((y_l_minus_1_shape[0], y_l_minus_1_shape[1]))
	# в зависимости от типа операции меняется основная формула функции
	if conv_params['convolution']:
		g = 1 # операция конволюции
	else:
		g = -1 # операция корреляции
	for i in range(dEdy_l_minus_1.shape[0]):
		for j in range(dEdy_l_minus_1.shape[1]):
			result = 0
			# матрица для демонстрации конволюции
			demo = np.zeros([dEdx_l.shape[0], dEdx_l.shape[1]])
			for i_x_l in range(dEdx_l.shape[0]):
				for j_x_l in range(dEdx_l.shape[1]):
					# перевод индексов в "нормальные" для извлечения элементов из матрицы w_l
					a = g*i_x_l*stride - g*i
					b = g*j_x_l*stride - g*j
					# проверка на вхождение в диапазон индексов ядра свертки
					if a in indexes_a and b in indexes_b:
						a = indexes_a.index(a)
						b = indexes_b.index(b)
						result += dEdx_l[i_x_l][j_x_l] * w_l[a][b]
						demo[i_x_l][j_x_l] = w_l[a][b]
			dEdy_l_minus_1[i][j] = result
			# вывод матрицы demo для отслеживания хода свертки
			# print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
	return dEdy_l_minus_1

def conv_weights_init(shape, quantity, weights_name, dir_npy):
	try:
		weights_matrices = np.load(dir_npy, allow_pickle=True).item().get(weights_name)
		print('веса для', weights_name, 'подгружены', len(weights_matrices)*weights_matrices[0].size)
	except:
		weights_matrices = []
		for i in range(quantity):
			weights_matrices.append(2 * np.random.random(shape) - 1)
		print('веса для', weights_name, 'созданы', len(weights_matrices)*weights_matrices[0].size)
	return weights_matrices

def fc_weights_init(shape, weights_name, dir_npy):
	try:
		weights_matrix = np.load(dir_npy, allow_pickle=True).item().get(weights_name)
		print('веса для', weights_name, 'подгружены', weights_matrix.size)
	except:
		weights_matrix = 2 * np.random.random(shape) - 1
		print('веса для', weights_name, 'созданы', weights_matrix.size)
	return weights_matrix

def convolution_feed(y_l_minus_1, w_l, w_l_name, w_shape_l, b_l, b_l_name, feature_maps, act_fn, dir_npy, conv_params):
	x_l = []
	y_l = []
	if not w_l:
		# инициализация w_l (количество ядер свертки равно число входов умножить на количество выходов)
		w_l = conv_weights_init(shape=w_shape_l, quantity=feature_maps*len(y_l_minus_1), weights_name=w_l_name, dir_npy=dir_npy)
	for i in range(len(y_l_minus_1)): # для всех y_l_minus_1
		for j in range(i*feature_maps, (i + 1)*feature_maps):
			# для каждой y_l_minus_1 функция конволюции вызывается feature_maps раз для создания "промежуточных" x_l
			x_l.append(convolution_feed_x_l(y_l_minus_1=y_l_minus_1[i], w_l=w_l[j], conv_params=conv_params))
	if len(b_l) == 0:
		# инициализация b_l (количество b_l равно числу выходов)
		b_l = conv_weights_init(shape=(1,1), quantity=feature_maps, weights_name=b_l_name, dir_npy=dir_npy)
	x_l_final = []
	for i in range(feature_maps): # итерация по количеству выходов
		x_l_final.append(0)
		for j in range(len(y_l_minus_1)): # итерация по количеству входных каналов
			x_l_final[-1] += x_l[j*feature_maps + i] # "финальный" x_l_final является суммой "промежуточных" x_l, полученных с каждой y_l_minus_1
		x_l_final[-1] += b_l[len(x_l_final)-1] # к x_l_final прибавляем соответствующий ему bias
		y_l.append(activation_fn(x_l_final[-1], fn_name=act_fn, feed=True)) # функция активации
	return y_l, w_l, b_l

def fc_multiplication(y_l_minus_1, w_l, w_l_name, b_l, b_l_name, neurons, act_fn, dir_npy):
	if w_l.size == 0:
		w_l = fc_weights_init(shape=(y_l_minus_1.shape[1], neurons), weights_name=w_l_name, dir_npy=dir_npy)
		b_l = fc_weights_init(shape=(1, neurons), weights_name=b_l_name, dir_npy=dir_npy)
	x_l = np.dot(y_l_minus_1, w_l) + b_l
	y_l = activation_fn(x_l, fn_name=act_fn, feed=True)
	return y_l, w_l, b_l

def activation_fn(matix, fn_name, feed):
	output_matix = np.copy(matix)
	if feed:
		if fn_name == 'sigmoid':
			output_matix = 1 / (1+np.exp(-output_matix)) # похоже, сообщение об ошибке можно проигнорировать https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
		if fn_name == 'relu':
			output_matix[output_matix<0] = 0
		if fn_name == 'softmax':
			output_matix = np.exp(output_matix) / np.exp(output_matix).sum()
	else:
		if fn_name == 'sigmoid':
			output_matix = output_matix * (1 - output_matix)
		if fn_name == 'relu': # relu для backprop рассчитывается исходя из того, что на вход подается y_l, а не x_l
			# output_matix[output_matix<=0] = 0 # после relu в матрице y_l не существует отрицательных значений
			output_matix[output_matix>0] = 1
		if fn_name == 'softmax':
			input_matix = np.copy(matix)
			output_matix = np.zeros((matix.shape[1], matix.shape[1]))
			for i in range(output_matix.shape[1]):
				for j in range(output_matix.shape[1]):
					if i==j:
						output_matix[i][i] = input_matix[0][i]*(1 - input_matix[0][i])
					else:
						output_matix[i][j] = - input_matix[0][i]*input_matix[0][j]
	return output_matix

def loss_fn(y_ground_truth, y_predicted, feed):
	if feed:
		# error_matix = (1/2)*(y_ground_truth - y_predicted)**2 # заменил на cross-entropy
		error_matix = - y_ground_truth * np.log(y_predicted)
	else:
		# error_matix = y_predicted - y_ground_truth
		error_matix = - (y_ground_truth/y_predicted)
	return error_matix

def matrix2vector(matrix):
	# функция объединения матриц в вектор
	vector = np.array([[]])
	for i in range(len(matrix)):
		reshaped_matrix = np.reshape(matrix[i], (1, matrix[i].shape[0]*matrix[i].shape[1]))
		vector = np.hstack((vector, reshaped_matrix))
	return vector

def vector2matrix(vector, matrix_shape):
	# функция разбиения вектора на матрицы
	matrices = []
	matrix_size = matrix_shape[0]*matrix_shape[1]
	for i in range(0, vector.size, matrix_size):
		matrix = np.reshape(vector[0][i:i+matrix_size], matrix_shape)
		matrices.append(matrix)
	return matrices

def matrix2vector_tf(matrix):
	# функция объединения матриц в вектор
	# объединение построено таким образом, чтобы можно было сравнить результаты с tensorflow
	matrices = []
	vector = np.array([])
	for i in range(len(matrix)):
		matrices.append(np.reshape(matrix[i], (1, matrix[i].size)))
	for i in range(matrices[0].size): # матрица все одного размера, выбираем любую
		for j in range(len(matrices)):
			vector = np.hstack((vector, matrices[j][0][i]))
	vector = np.reshape(vector, (1, vector.size))
	return vector

def vector2matrix_tf(vector, matrix_shape):
	# функция разбиения вектора на матрицы
	# разбиение построено таким образом, чтобы можно было сравнить результаты с tensorflow
	matrices = []
	for i in range(int(vector.size/(matrix_shape[0]*matrix_shape[1]))):
		matrices.append(np.array([]))
	j = 0
	for i in range(vector.size):
		matrices[j] = np.hstack((matrices[j], vector[0][i]))
		j += 1
		if j == len(matrices):
			j = 0
	for i in range(len(matrices)):
		matrices[i] = np.reshape(matrices[i], matrix_shape)
	return matrices

def fc_backpropagation(y_l_minus_1, dEdy_l, y_l, w_l, b_l, act_fn, alpha):
	# вычисление dE/dx_l, то есть backprop через функцию активации
	if act_fn == 'softmax':
		dEdx_l = np.dot(dEdy_l, activation_fn(y_l, fn_name=act_fn, feed=False))
	else:
		dEdx_l = dEdy_l * activation_fn(y_l, fn_name=act_fn, feed=False)
	# вычисление частных производных
	dEdw_l = np.dot(y_l_minus_1.T, dEdx_l)
	dEdb_l = dEdx_l
	dEdy_l_minus_1 = np.dot(dEdx_l, w_l.T)
	# обновление матриц весов
	w_l = w_l - alpha * dEdw_l
	b_l = b_l - alpha * dEdb_l
	return dEdy_l_minus_1, w_l, b_l

def convolution_backpropagation(y_l_minus_1, y_l, w_l, b_l, dEdy_l, feature_maps, act_fn, alpha, conv_params):
	list_of_dEdy_l_minus_1 = []
	list_of_dEdx_l = []
	for i in range(len(y_l)):
		# сначала происходит расчет dEdx_l, то есть обратное прохождение dEdy_l через функцию активации
		list_of_dEdx_l.append(dEdy_l[i] * activation_fn(y_l[i], fn_name=act_fn, feed=False))
		# вследствие того, что только одна b_l приходится на одну карту признаков, то dEdb_l является суммой по всем элементам dEdx_l
		dEdb_l = list_of_dEdx_l[-1].sum()
		b_l[i] = b_l[i] - alpha * dEdb_l # обновление b_l
	for i in range(len(y_l_minus_1)): # итерация по входным каналам
		dEdy_l_minus_1 = 0
		k = 0
		# далее итерация по "промежуточным" картам признаков, соответствующим i-тому входному каналу
		# сами "промежуточные" карты не используются! в вычислениях присутствуют только "финальные" карты
		# при этом количество "финальных" карт (здесь dEdx_l) равно feature_maps, тогда как количество w_l равно feature_maps*y_l_minus_1
		# отсюда использование дополнительного "итератора" k, таким образом к dEdx_l мы обращаемся с помощью k, а к w_l с помощью j
		for j in range(i*feature_maps, (i + 1)*feature_maps):
			dEdw_l = convolution_back_dEdw_l(
				y_l_minus_1=y_l_minus_1[i], # i-тый входной канал
				w_l_shape=w_l[j].shape, # j-тый w_l
				dEdx_l=list_of_dEdx_l[k], # k-тый dEdx_l
				conv_params=conv_params
			)
			# через слой y_l_minus_1 проходят суммы показателей со всех карт признаков
			dEdy_l_minus_1 += convolution_back_dEdy_l_minus_1(
				dEdx_l=list_of_dEdx_l[k], # k-тый dEdx_l
				w_l=w_l[j], # j-тый w_l
				y_l_minus_1_shape=y_l_minus_1[i].shape, # i-тый входной канал
				conv_params=conv_params
			)
			w_l[j] = w_l[j] - alpha * dEdw_l # обновление w_l
			k += 1
		list_of_dEdy_l_minus_1.append(dEdy_l_minus_1)
	return list_of_dEdy_l_minus_1, w_l, b_l
