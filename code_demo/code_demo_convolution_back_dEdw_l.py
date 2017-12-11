import numpy as np

w_l_shape = (2,2)

# если stride = 1
dEdx_l = np.array([
	[1,2,3,4],
	[5,6,7,8],
	[9,10,11,12],
	[13,14,15,16]])

# если stride = 2 и 'convolution':False (при конволюции и кросс-корреляци x_l получаются разного размера)
# dEdx_l = np.array([
# 	[1,2],
# 	[3,4]])

# если stride = 2 и 'convolution':True
# dEdx_l = np.array([
# 	[1,2,3],
# 	[4,5,6],
# 	[7,8,9]])

y_l_minus_1 = np.zeros((4,4))

other_parameters={
	'convolution':True,
	'stride':1,
	'center_w_l':(0,0)
}

def convolution_back_dEdw_l(y_l_minus_1, w_l_shape, dEdx_l, conv_params):
	indexes_a, indexes_b = create_indexes(size_axis=w_l_shape, center_w_l=conv_params['center_w_l'])
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
			print('a=' + str(a) + '; b=' + str(b) + '\n', demo)
	return dEdw_l

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

print(convolution_back_dEdw_l(y_l_minus_1, w_l_shape, dEdx_l, other_parameters))
