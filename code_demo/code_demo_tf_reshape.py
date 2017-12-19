import numpy as np
import tensorflow as tf

tf_w = tf.truncated_normal([2, 2, 1, 4], stddev=0.1)

with tf.Session() as sess:
	np_w = sess.run(tf_w)

print('\n Так выглядят веса tensorflow: \n \n', np_w)
print('\n \n Простая итерация по тензору не дает нужного результата:')
for i in range(len(np_w)):
	print('\n', np_w[i])

conv_w = []
np_w = np.reshape(np_w, (np_w.size,))
np_w = np.reshape(np_w, (4,2,2), order='F')
print('\n \n А вот матрицы в том виде, как их извлекат tensorflow:')
for i in range(4):
	conv_w.append(np_w[i].T)
	print('\n', conv_w[-1])