import numpy as np

y_l = np.array([
	[1,0,2,3],
	[4,6,6,8],
	[3,1,1,0],
	[1,2,2,4]])

other_parameters={
	'convolution':False,
	'stride':2,
	'center_window':(0,0),
	'window_shape':(2,2)
}

def maxpool(y_l, conv_params):
	indexes_a, indexes_b = create_indexes(size_axis=conv_params['window_shape'], center_w_l=conv_params['center_window'])
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

out_maxpooling = maxpool(y_l, other_parameters)
print('выходная матрица:', '\n', out_maxpooling[0])
print('\n', 'матрица с координатами для backprop:', '\n', out_maxpooling[1])
