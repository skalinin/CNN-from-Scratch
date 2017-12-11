import numpy as np

w_l = np.array([
	[1,2],
	[3,4]])

# если stride = 1
dEdx_l = np.zeros((3,3))

# если stride = 2 и 'convolution':False (при конволюции и кросс-корреляци x_l могут получиться разного размера)
# dEdx_l = np.zeros((2,2))

# если stride = 2 и 'convolution':True
# dEdx_l = np.zeros((2,2))

y_l_minus_1_shape = (3,3)

other_parameters={
	'convolution':True,
	'stride':1,
	'center_w_l':(0,0)
}

def convolution_back_dEdy_l_minus_1(dEdx_l, w_l, y_l_minus_1_shape, conv_params):
	indexes_a, indexes_b = create_indexes(size_axis=w_l.shape, center_w_l=conv_params['center_w_l'])
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
			print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
	return dEdy_l_minus_1

def create_axis_indexes(size_axis, center_w_l):
	coordinates = []
	for i in range(-center_w_l, size_axis-center_w_l):
		coordinates.append(i)
	return coordinates

def create_indexes(size_axis, center_w_l):
	# расчет координат на осях ядра свертки в зависимости от номера центрального элемента ядра
	coordinates_a = create_axis_indexes(size_axis=size_axis[0], center_w_l=center_w_l[0])
	coordinates_b = create_axis_indexes(size_axis=size_axis[1], center_w_l=center_w_l[1])
	return coordinates_a, coordinates_b

print(convolution_back_dEdy_l_minus_1(dEdx_l, w_l, y_l_minus_1_shape, other_parameters))
