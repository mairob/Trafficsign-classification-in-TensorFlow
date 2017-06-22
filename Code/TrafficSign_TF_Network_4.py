import numpy as np
import tensorflow as tf


#Note: this one should give you around 99.1% on an augumented GTSRB

image_size = 48  #width = height
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
#Note: dropout should be around 35...50%
#Note: learningrate should be around 1e-4....1e-6
with tf.name_scope('Network'):

	with tf.name_scope('input'):
		x_image = tf.placeholder(tf.float32, [None, image_size, image_size, col_channels], name='Images_raw')
		y_raw = tf.placeholder(tf.int32, [None] ,name='Labels_raw')
		y_= tf.one_hot(indices=y_raw, depth=43 ,  name='Labels_oneHot')
		
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)

	with tf.name_scope('learningrate'):
		learningrate = tf.placeholder(tf.float32)

	with tf.name_scope('Layer1'):
		W_conv1 = initVariable("W1", [5, 5, 3, 96]) 
		b_conv1 = initVariable("B1", [96])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(tf.nn.local_response_normalization(h_conv1))  #resulting feature maps = 24x24 Pixel


	with tf.name_scope('Layer2'):
		W_conv2 = initVariable("W2",[3, 3, 96, 96])
		b_conv2 = initVariable("B2",[96])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(tf.nn.local_response_normalization(h_conv2)) #resulting feature maps = 12x12 Pixel


	with tf.name_scope('Layer3'):
		W_conv3 = initVariable("W3",[3, 3, 96, 48])
		b_conv3 = initVariable("B3",[48])
		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
		h_conv3_flat = tf.reshape(h_conv3, [-1, 12*12*48]) 

	with tf.name_scope('Layer4'):
		W_conv4 = initVariable("W4",[3, 3, 48, 48]) 
		b_conv4 = initVariable("B4",[48])
		h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

	with tf.name_scope('Layer5'):
		W_conv5 = initVariable("W5",[3, 3, 48, 48]) 
		b_conv5= initVariable("B5",[48]) 
		h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
		h_pool5 = max_pool_2x2(tf.nn.local_response_normalization(h_conv5)) #resulting feature maps = 6x6 Pixel
		
	with tf.name_scope('Layer6'): 
		W_conv6 = initVariable("W6",[2, 2, 48, 48]) 
		b_conv6 = initVariable("B6",[48])
		h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
		h_pool6= max_pool_2x2(tf.nn.local_response_normalization(h_conv6)) #resulting feature maps = 3x3 Pixel
		h_pool6_flat = tf.reshape(h_pool6, [-1, 3*3*48]) 

	with tf.name_scope('Concat'):
		ccat = tf.concat([h_conv3_flat, h_pool5_flat], 1)	
		W_ccat = initVariable("Wccat",[7344 , 2048])  #7344 = 12*12*48 +3*3*48
		b_ccat = initVariable("Bccat",[2048])
		h_ccat = tf.nn.relu(tf.matmul(ccat, W_ccat) + b_ccat)
		h_ccat_drop = tf.nn.dropout(h_ccat, keep_prob)

		
	with tf.name_scope('Fully1'):
		W_fc1 = initVariable("Wfc1",[3*3*48 , 1024])
		b_fc1 = initVariable("Bfc1",[1024])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	with tf.name_scope('Fully2'):
		fc2_ccat = tf.concat([h_ccat_drop, h_fc1_drop], 1)	
		W_fc2 = initVariable("Wfc2",[3072 , 768]) 
		b_fc2 = initVariable("Bfc2",[768])
		h_fc2 = tf.nn.relu(tf.matmul(fc2_ccat, W_fc2) + b_fc2)
		h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


	with tf.name_scope('OutputLayer'):
		W_out = initVariable("Wout",[768, 43])
		b_out = initVariable("Bout",[43])

	with tf.name_scope("softmax"):
		y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_out) + b_out)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
