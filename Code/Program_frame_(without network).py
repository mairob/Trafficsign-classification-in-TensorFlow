import os
import sys
import random
import csv
import numpy as np
import tensorflow as tf
import gc
import cv2
import skimage 


#supress warnings for "CPU- Instructionset not optimized"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_size = 48  #width = height
col_channels = 3 	#RGB

logs_path = r'C:\Users\......\LogginFolder'

trng_path_file = r'C:\Users\.....\Training.csv'
trng_path_folder = r'C:\Users\.....\TrainingFolder'

equ_path_file = r'C:\Users\.....\Equalized.csv'
equ_path_folder = r'C:\Users\.....\EqualizedFolder'

test_path_file = r'C:\Users\.....\Test.csv'
test_path_folder = r'C:\Users\.....\TestFolder'




def normalizeImages(lst_images):
	"""
    Preprocessing images by normalizing
    """
	return [((x - x.mean())/x.std()) for x in lst_images]
	

def loadDataSet(csv_path, images_rootpath):
	"""
    Loading function... well..somehow we need to import the data
	Images are already preproccessed, hence the different folder structure
		-> see corressponding file for more information
    """
	images = [] 
	for c in range(0,43):
		curFileCnt = next(os.walk(images_rootpath +'/' + str(c)))[2] 
		for cnt in range(len(curFileCnt)):	
			curFilePath = images_rootpath + '/' + str(c) + '/GTSRB_' + str(c) + '_' + str(cnt) + '.png'
			images.append(cv2.imread(curFilePath,1))
			
	labels = [] 
	gtFile = open(csv_path) 
	gtReader = csv.reader(gtFile, delimiter="\n") 
	for row in gtReader:
		labels += row
	gtFile.close()

	return normalizeImages(images), list(map(int, labels))	

	
def convertToUINT8(x):
	"""
    Loading function... well..somehow we need to import the data
	Resize images  to your desire.
    """
	return skimage.img_as_ubyte(x / np.amax(x))



def variable_summaries(var, summary_name):
	"""
    Helper function for visualizing scalars in TensorBoard
    """
	with tf.name_scope("Summary_" + str(summary_name)):
		tf.summary.scalar("raw", var)
		tf.summary.histogram('histogram', var)
		
	
def VisualizeConvolutions(myTensor, sizeInfo,  name):
	"""
    Helper function for visualizing feature maps in TensorBoard (image tab)
    """
	V = tf.slice(myTensor, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_' + name)
	V = tf.reshape(V, (sizeInfo[0], sizeInfo[1], sizeInfo[2]))
	V = tf.transpose(V, (2, 0, 1))
	V = tf.reshape(V, (-1, sizeInfo[0], sizeInfo[1], 1))
	tf.summary.image(name, V,  max_outputs=4)	
	
	
	
def evaluateTestSet(mySession):
	"""
    Used for evaluation of test set after each epoch
	Testset is split in batches due to GPU -RAM limitations
	
    """
	mean_acc = 0.
	splitfactor = 51
	batch_size =  len(test_images)//splitfactor

	for i in range(splitfactor+1):
		upper_limit = (i+1)*batch_size
		if upper_limit >= len(test_images):
			upper_limit = len(test_images)

		batch_xs = [test_images[i] for i in range(i*batch_size, upper_limit)]
		batch_ys = [test_labels[i] for i in range(i*batch_size, upper_limit)]

		mean_acc += accuracy.eval(session=mySession,feed_dict={x_image: batch_xs, y_raw: batch_ys, keep_prob: 1.0, learningrate: 1e-4})


	return mean_acc/ (splitfactor+ 1)
	
	
	
	
#######################################################
#######################################################	

	
print(">>>> Load traing data")
trng_images, trng_labels = loadDataSet(trng_path_file, trng_path_folder)


print(">>>> Load test data")
test_images, test_labels = loadDataSet(test_path_file, test_path_folder)



#######################################################
#######################################################

								####################
								#### INSERT NETWORK ####
								####################
								
								
	
	#Note: add the following for visualization

	with tf.name_scope('Visualizing_RawImage'):
			img_raw = tf.slice(x_image, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_raw_image')
			tf.summary.image("raw_image", img_raw,  max_outputs=32)

			
	with tf.name_scope('Visualizing_Conv'):
		savingPath = tf.placeholder(tf.string)
		index = tf.placeholder(tf.int32)

		VisualizeConvolutions(h_conv1, [image_size, image_size, 64], "Conv1") #64 =  nr . of resulting feature maps from conv.
		
		
	with tf.name_scope('Visualizing_Msc'):
		variable_summaries(cross_entropy, 'crossentropy')
		variable_summaries(accuracy, 'TrainingAcc')	
		
		#Don't forget to merge all summaries !!!
		summary_op = tf.summary.merge_all()



#######################################################
#######################################################


print(">>>> Create session")

with tf.Session() as sess:

	#Initialize network and set up logging for TensorBoard
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(logs_path)
	summary_writer.add_graph(sess.graph)

	
	print(">>>> Start training on training set")	

	#Note: Outer loop defines nr. of epochs to train
	#Note: Inner loop defines nr. of trainingsteps  so that: 
	#		    nr. of step * batch_size = nr. of images in current dataset
	
	batch_size = 256
	
	for epoch in range(200):
	
		for i in range(1200):
			sample_index = random.sample(range(len(trng_images)), batch_size) 
			batch_xs = [trng_images[i] for i in sample_index]
			batch_ys = [trng_labels[i] for i in sample_index]
			
			#Adjust dropout and learningrate after a given time
			if epoch < 150:	
				_, summary = sess.run([train_step, summary_op], feed_dict={x_image: batch_xs, y_raw: batch_ys, keep_prob: 0.65, learningrate: 1e-4})
				summary_writer.add_summary(summary, i)	
				
			else:
				_, summary = sess.run([train_step, summary_op], feed_dict={x_image: batch_xs, y_raw: batch_ys, keep_prob: 0.6, learningrate: 1e-5})
				summary_writer.add_summary(summary, i)	
	
		print(">UPDATE< Accuracy in TEST-set after epoch %i : %.4f " %(epoch+1, evaluateTestSet(mySession=sess)))

		
		
		
	print(">>>> Clear RAM from training data ")
	del trng_images
	del trng_labels
	gc.collect()	
	
	
	print(">>>> Load equalized training set")
	equ_images, equ_labels = loadDataSet(equ_path_file, equ_path_folder)

	
	print(">>>> Start training on equalized set")	

	#Note: Outer loop defines nr. of epochs to train
	#Note: Inner loop defines nr. of trainingsteps  so that: 
	#		    nr. of step * batch_size = nr. of images in current dataset
	
	batch_size = 128
	
	for epoch in range(200):
	
		for i in range(67):
			sample_index = random.sample(range(len(equ_images)), batch_size) 
			batch_xs = [equ_images[i] for i in sample_index]
			batch_ys = [equ_labels[i] for i in sample_index]
			
			#Adjust dropout and learningrate after a given time
			if epoch < 150:	
				_, summary = sess.run([train_step, summary_op], feed_dict={x_image: batch_xs, y_raw: batch_ys, keep_prob: 0.55, learningrate: 1e-5})
				summary_writer.add_summary(summary, i)	
				
			else:
				_, summary = sess.run([train_step, summary_op], feed_dict={x_image: batch_xs, y_raw: batch_ys, keep_prob: 0.5, learningrate: 1e-6})
				summary_writer.add_summary(summary, i)	

		
		print(">UPDATE< Accuracy in TEST-set after epoch %i  on equalized set : %.4f " %(epoch+1, evaluateTestSet(mySession=sess)))

