import numpy as np # основная библиотека для работы с массивами
import matplotlib.pyplot as plt # для построения графиков
import PIL # для работы с изображениями
import os # для работы с файлами на диске
import model # книга с функциями
from tensorflow.examples.tutorials.mnist import input_data # датасет mnist
import tensorflow as tf
# from pudb import set_trace; set_trace() # для дебага

# закрепление сидов
np.random.seed(0)
tf.set_random_seed(0)

train_model = False # обучение или тест модели

# загрузка датасета
image_storage = [] # здесь будут храниться изображения
truth_storage = [] # здесь будут храниться ground truth лейблы для изображений
dir_storage = [] # директории к изображениям
weight_dir = './cnn_weights_mnist.npy'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if train_model:
	mnist_dataset = mnist.train.images
	truth_storage = mnist.train.labels
else:
	for i in range(55000): # mnist.test сортируется после прохождения по mnist.train (для сравнения результатов с tensorflow)
		batch = mnist.train.next_batch(1)
	mnist_dataset = mnist.test.images
	truth_storage = mnist.test.labels

for input_image in mnist_dataset:
	image_storage.append(np.reshape(input_image, (28, 28))) # (784,1) -> (28,28)
# for i in range(len(mnist_dataset)): # сборка image_storage через next_batch с фиксированным сидом для сравнения результатов с моделью на tensroflow
# 	input_image = mnist.train.next_batch(1)
# 	image_storage.append(np.reshape(input_image[0], (28, 28))) # (784,1) -> (28,28)
# 	truth_storage.append(np.reshape(input_image[1], (10,))) # (1,10) -> (10,)
dir_storage = ['None' for i in range(len(image_storage))]

# первый и последний шаги
if train_model:
	start_step = model.get_start_step(weight_dir)
	end_step = len(image_storage)
	len_dataset = 1000 # частота вывода print и сохранения весов (не менять при возобновлении обучения)
else:
	start_step = 0
	end_step = len(image_storage)
	len_dataset = 1000

# перемешивание датасета приводит к тому, что изображения становятся в последовательность, аналогичную next_batch(1) для данного сида
image_storage, truth_storage, dir_storage = model.shuffle_list(image_storage, truth_storage, dir_storage)

# параметры сети
model_settings = {
	'learning_rate':0.01, # коэффициент обучения
	'conv_shape_1':(2,2), # размер ядра свертки
	'conv_shape_2':(3,3),
	'maxpool_shape_1':(2,2), # размер окна макспулинга
	'conv_feature_1':5, # количесвто feature maps на выходе функции
	'conv_feature_2':20,
	'conv_stride_1':2, # величина шага
	'conv_stride_2':1,
	'maxpool_stride_1':2,
	'fc_neurons_1':2000, # количество нейронов в скрытом слое
	'conv_fn_1':'relu', # функция активации
	'conv_fn_2':'sigmoid',
	'fc_fn_1':'sigmoid',
	'fc_fn_2':'softmax',
	'conv_conv_1':False, # операция конволюции или кросс-корреляции
	'conv_conv_2':False,
	'maxpool_conv_1':False, # "конволюция" или "корреляция" для операции макспулинга
	'conv_center_1':(0,0), # центр ядра
	'conv_center_2':(1,1),
	'maxpool_center_1':(0,0)
}

# параметры для первого слоя конволюции (начальные параметры будут инициализированы во время работы сети)
# веса для дообучения сети будут подгружены из файла
conv_w_1 = []
conv_b_1 = []
# параметры для второго слоя конволюции
conv_w_2 = []
conv_b_2 = []
# параметры для первого слоя fc-сети
fc_w_1 = np.array([[]])
fc_b_1 = np.array([[]])
# параметры для второго слоя fc-сети
fc_w_2 = np.array([[]])
fc_b_2 = np.array([[]])

# создание начальных значений с закрепленным сидом для сравнения результатов с моделью на tensorflow
# этот участок кода можно просто закомментировать, если нет необходимости в инициализации tensorflow-весов
if not os.path.isfile(weight_dir):
	tf_w1 = tf.truncated_normal([2, 2, 1, 5], stddev=0.1)
	tf_w2 = tf.constant(0.1, shape=[5])
	tf_w3 = tf.truncated_normal([3, 3, 5, 20], stddev=0.1)
	tf_w4 = tf.constant(0.1, shape=[20])
	tf_w5 = tf.truncated_normal([7*7*20, 2000], stddev=0.1)
	tf_w6 = tf.constant(0.1, shape=[2000])
	tf_w7 = tf.truncated_normal([2000, 10], stddev=0.1)
	tf_w8 = tf.constant(0.1, shape=[10])
	with tf.Session() as sess:
		w1, w2, w3, w4, w5, w6, w7, w8 = sess.run([tf_w1, tf_w2, tf_w3, tf_w4, tf_w5, tf_w6, tf_w7, tf_w8])
	w1 = np.reshape(w1, (w1.size,))
	w1 = np.reshape(w1, (5,2,2), order='F')
	for i in range(5):
		conv_w_1.append(w1[i].T)
	w3 = np.reshape(w3, (w3.size,))
	w3 = np.reshape(w3, (5*20,3,3), order='F')
	for i in range(5*20):
		conv_w_2.append(w3[i].T)
	conv_b_1 = w2
	conv_b_2 = w4
	fc_w_1 = w5
	fc_b_1 = w6
	fc_w_2 = w7
	fc_b_2 = w8

# загрузка результатов предыдущего обучения из дампов модели (если первое обучение - создаются пустые листы)
if train_model:
	loss_change = model.get_saved('loss_change', weight_dir)
	accuracy_change = model.get_saved('accuracy_change', weight_dir)
else:
	loss_change = []
	accuracy_change = []
for step in range(start_step, end_step):
	# извлечение изображения из хранилища
	image_id = step%len(image_storage) # на каждом шаге обновляются веса для одного изображения
	print ('до вывода результатов', str(round((step%len_dataset)*100/len_dataset)) + '%', end="\r")
	input_image = [image_storage[image_id]] # здесь лист, так как convolution_feed на вход принимает лист, состоящий из feature maps
	y_true = truth_storage[image_id]
	# прямое прохожение сети
	# первый конволюционный слой
	conv_y_1, conv_w_1, conv_b_1 = model.convolution_feed(
		y_l_minus_1=input_image,
		w_l=conv_w_1,
		w_l_name='conv_w_1', # для подгрузки весов из файла
		w_shape_l=model_settings['conv_shape_1'],
		b_l=conv_b_1,
		b_l_name='conv_b_1',
		feature_maps=model_settings['conv_feature_1'],
		act_fn=model_settings['conv_fn_1'],
		dir_npy=weight_dir,
		conv_params={
			'convolution':model_settings['conv_conv_1'],
			'stride':model_settings['conv_stride_1'],
			'center_w_l':model_settings['conv_center_1']
		}
	)
	# слой макспулинга
	conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = model.maxpool_feed(
		y_l=conv_y_1,
		conv_params={
			'window_shape':model_settings['maxpool_shape_1'],
			'convolution':model_settings['maxpool_conv_1'],
			'stride':model_settings['maxpool_stride_1'],
			'center_window':model_settings['maxpool_center_1']
		}
	)
	# второй конволюционный слой
	conv_y_2, conv_w_2, conv_b_2 = model.convolution_feed(
		y_l_minus_1=conv_y_1_mp,
		w_l=conv_w_2,
		w_l_name='conv_w_2',
		w_shape_l=model_settings['conv_shape_2'],
		b_l=conv_b_2,
		b_l_name='conv_b_2',
		feature_maps=model_settings['conv_feature_2'],
		act_fn=model_settings['conv_fn_2'],
		dir_npy=weight_dir,
		conv_params={
			'convolution':model_settings['conv_conv_2'],
			'stride':model_settings['conv_stride_2'],
			'center_w_l':model_settings['conv_center_2']
		}
	)
	# конвертация полученных feature maps в вектор
	conv_y_2_vect = model.matrix2vector_tf(conv_y_2)
	# первый слой fully connected сети
	fc_y_1, fc_w_1, fc_b_1 = model.fc_multiplication(
		y_l_minus_1=conv_y_2_vect,
		w_l=fc_w_1,
		w_l_name='fc_w_1',
		b_l=fc_b_1,
		b_l_name='fc_b_1',
		neurons=model_settings['fc_neurons_1'],
		act_fn=model_settings['fc_fn_1'],
		dir_npy=weight_dir
	)
	# второй слой fully connected сети
	fc_y_2, fc_w_2, fc_b_2 = model.fc_multiplication(
		y_l_minus_1=fc_y_1,
		w_l=fc_w_2,
		w_l_name='fc_w_2',
		b_l=fc_b_2,
		b_l_name='fc_b_2',
		neurons=len(y_true), # количество нейронов на выходе моледи равно числу классов
		act_fn=model_settings['fc_fn_2'],
		dir_npy=weight_dir
	)
	# ошибка модели
	fc_error = model.loss_fn(y_true, fc_y_2, feed=True)
	# сохранение значений loss и accuracy
	loss_change.append(fc_error.sum())
	accuracy_change.append(y_true.argmax() == fc_y_2.argmax())
	# обратное прохожение по сети
	if train_model:
		# backprop через loss-функцию
		dEdfc_y_2 = model.loss_fn(y_true, fc_y_2, feed=False)
		# backprop через второй слой fc-сети
		dEdfc_y_1, fc_w_2, fc_b_2 = model.fc_backpropagation(
			y_l_minus_1=fc_y_1,
			dEdy_l=dEdfc_y_2,
			y_l=fc_y_2,
			w_l=fc_w_2,
			b_l=fc_b_2,
			act_fn=model_settings['fc_fn_2'],
			alpha=model_settings['learning_rate']
		)
		# backprop через первый слой fc-сети
		dEdfc_y_0, fc_w_1, fc_b_1 = model.fc_backpropagation(
			y_l_minus_1=conv_y_2_vect,
			dEdy_l=dEdfc_y_1,
			y_l=fc_y_1,
			w_l=fc_w_1,
			b_l=fc_b_1,
			act_fn=model_settings['fc_fn_1'],
			alpha=model_settings['learning_rate']
		)
		# конвертация полученного вектора в feature maps
		dEdconv_y_2 = model.vector2matrix_tf(
			vector=dEdfc_y_0,
			matrix_shape=conv_y_2[0].shape # размерность одной из матриц feature map
		)
		# backprop через второй слой конволюции
		dEdconv_y_1_mp, conv_w_2, conv_b_2 = model.convolution_backpropagation(
			y_l_minus_1=conv_y_1_mp, # так как слой макспулинга!
			y_l=conv_y_2,
			w_l=conv_w_2,
			b_l=conv_b_2,
			dEdy_l=dEdconv_y_2,
			feature_maps=model_settings['conv_feature_2'],
			act_fn=model_settings['conv_fn_2'],
			alpha=model_settings['learning_rate'],
			conv_params={
				'convolution':model_settings['conv_conv_2'],
				'stride':model_settings['conv_stride_2'],
				'center_w_l':model_settings['conv_center_2']
			}
		)
		# backprop через слой макспулинга
		dEdconv_y_1 = model.maxpool_back(
			dEdy_l_mp=dEdconv_y_1_mp,
			y_l_mp_to_y_l=conv_y_1_mp_to_conv_y_1,
			y_l_shape=conv_y_1[0].shape
		)
		# backprop через первый слой конволюции
		dEdconv_y_0, conv_w_1, conv_b_1 = model.convolution_backpropagation(
			y_l_minus_1=input_image,
			y_l=conv_y_1,
			w_l=conv_w_1,
			b_l=conv_b_1,
			dEdy_l=dEdconv_y_1,
			feature_maps=model_settings['conv_feature_1'],
			act_fn=model_settings['conv_fn_1'],
			alpha=model_settings['learning_rate'],
			conv_params={
				'convolution':model_settings['conv_conv_1'],
				'stride':model_settings['conv_stride_1'],
				'center_w_l':model_settings['conv_center_1']
			}
		)
	# вывод результатов
	if len(loss_change)%len_dataset == 0:
		print('шаг:', len(loss_change), 'loss:', sum(loss_change[-len_dataset:])/len_dataset, 'accuracy:', sum(accuracy_change[-len_dataset:])/len_dataset)
		# сохранение весов
		if train_model:
			np.save(weight_dir, {
				'step':step,
				'loss_change':loss_change,
				'accuracy_change':accuracy_change,
				'conv_w_1':conv_w_1,
				'conv_b_1':conv_b_1,
				'conv_w_2':conv_w_2,
				'conv_b_2':conv_b_2,
				'fc_w_1':fc_w_1,
				'fc_b_1':fc_b_1,
				'fc_w_2':fc_w_2,
				'fc_b_2':fc_b_2
				}
			)
	# перемешивание датасета [https://stats.stackexchange.com/questions/272409/mixing-shuffle-order-on-training-set-for-future-epochs]
	if train_model and len(loss_change)%len(image_storage) == 0:
		image_storage, truth_storage, dir_storage = model.shuffle_list(image_storage, truth_storage, dir_storage)

if not train_model:
	print('test_loss:', sum(loss_change)/len(loss_change), 'test_accuracy:', sum(accuracy_change)/len(accuracy_change))
