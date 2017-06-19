import numpy as np
import tensorflow as tf

image_size = 48   #width = height
col_channels = 3 	#RGB


def initVariable(name, shape):
	"""
    Initialize weights and biases based on http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, W):
	"""
    Basic convolution. 
    """
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""
    Basic pooling
    """
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
						

						
#Note:  tf.name_scope is used for structuring the graph
#Note: learningrate should be around 1e-4....1e-6
#Note: Minimum viable model
with tf.name_scope('Network'):

	with tf.name_scope('input'):
		x_image = tf.placeholder(tf.float32, [None, image_size, image_size, col_channels], name='Images_raw')
		y_raw = tf.placeholder(tf.int32, [None] ,name='Labels_raw')
		y_= tf.one_hot(indices=y_raw, depth=43 ,  name='Labels_oneHot')
		
	with tf.name_scope('learningrate'):
		learningrate = tf.placeholder(tf.float32)

	with tf.name_scope('OutputLayer'):
		x_image_flat = tf.reshape(x_image, [-1, image_size * image_size *col_channels])
		W_out = initVariable("Wout",[image_size * image_size *col_channels, 43])
		b_out = initVariable("Bout", [43])

	with tf.name_scope("softmax"):
		y_conv=tf.nn.softmax(tf.matmul(x_image_flat, W_out) + b_out)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)


	with tf.name_scope('Accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
