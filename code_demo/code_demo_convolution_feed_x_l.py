import numpy as np

# w_l = np.array([
# 	[1,2,3,4],
# 	[5,6,7,8],
# 	[9,10,11,12],
# 	[13,14,15,16]])

w_l = np.array([
	[1,2],
	[3,4]])

y_l_minus_1 = np.zeros((3,3))

other_parameters={
	'convolution':True,
	'stride':1,
	'center_w_l':(0,0)
}

def convolution_feed_x_l(y_l_minus_1, w_l, conv_params):
	indexes_a, indexes_b = create_indexes(size_axis=w_l.shape, center_w_l=conv_params['center_w_l'])
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
				print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
	return x_l

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

print(convolution_feed_x_l(y_l_minus_1, w_l, other_parameters))
