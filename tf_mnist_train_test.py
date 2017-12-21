import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# from pudb import set_trace; set_trace() # для дебага

np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# генерация матриц отдельно от основных переменных для закрепления случайных значений и сравнения результатов с моделью на numpy
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

x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
input_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = tf.Variable(w1)
b_conv1 = tf.Variable(w2)
h_conv1 = tf.nn.relu(tf.nn.conv2d(input_image, w_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_conv2 = tf.Variable(w3)
b_conv2 = tf.Variable(w4)
h_conv2 = tf.nn.sigmoid(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*20])

w_fc1 = tf.Variable(w5)
b_fc1 = tf.Variable(w6)
h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, w_fc1) + b_fc1)

w_fc2 = tf.Variable(w7)
b_fc2 = tf.Variable(w8)
y_conv = (tf.matmul(h_fc1, w_fc2) + b_fc2) # softmax применяется в loss

# loss = (1/2)*tf.reduce_sum(tf.square(y_true-y_conv))
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))
# loss = -tf.reduce_sum(y_true * tf.log(y_conv), 1) # в этом случае на последнем слое нужно добавить tf.nn.softmax
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

len_dataset = 1000
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	accuracy_change = []
	loss_change = []
	for i in range(55000):
		batch = mnist.train.next_batch(1)
		train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x:batch[0], y_true:batch[1]})
		train_step.run(feed_dict={x:batch[0], y_true:batch[1]})
		loss_change.append(train_loss)
		accuracy_change.append(train_accuracy)
		if len(loss_change)%len_dataset == 0:
			print('шаг:', len(loss_change), 'loss:', sum(loss_change[-len_dataset:])/len_dataset, 'accuracy:', sum(accuracy_change[-len_dataset:])/len_dataset)

	print('mnist test...')
	accuracy_change = []
	loss_change = []
	for i in range(10000):
		batch = mnist.test.next_batch(1)
		train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x:batch[0], y_true:batch[1]})
		loss_change.append(train_loss)
		accuracy_change.append(train_accuracy)
		if len(loss_change)%len_dataset == 0:
			print('шаг:', len(loss_change), 'loss:', sum(loss_change[-len_dataset:])/len_dataset, 'accuracy:', sum(accuracy_change[-len_dataset:])/len_dataset)
	print('test_loss:', sum(loss_change)/len(loss_change), 'test_accuracy:', sum(accuracy_change)/len(accuracy_change))
