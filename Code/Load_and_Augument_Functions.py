import os
import cv2
import csv
import numpy as np



#Warning: NO RESIZING IS DONE HERE. If you want to do that, I encourage you to do it
#		  right before saving the enhanced trng-set with something like: cv2.resize(img, (48,48))

#Note: Code may containg slight errors
#Note: Baisc functions for loading the original GTSRB
#Note: Functions for augumenting the data
#Note: Functions for making the aug. data easier accessible


def loadGTSRBdataset(csv_path, images_path):
	"""
    Adapted loading function for trng and test-set based on http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#CodesnippetsPython

    #Example for loading the original test dataset
	#csv_path = r'C:\Users\...\GTSRB\Test\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images\GT-final_test.csv'
	#images_path = r'C:\Users\...\GTSRB\Test\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images'

    """
	images = []
	labels = [] 
	gtFile = open(csv_path) 
	gtReader = csv.reader(gtFile, delimiter=';') 
	next(gtReader) 
	for row in gtReader:
	    images.append(cv2.cvtColor(cv2.imread(images_path + '/' + row[0]), cv2.COLOR_BGR2RGB))
	    labels.append(row[7]) 
	gtFile.close()
	return images, list(map(int, labels))



def enhance_and_save_trng(csv_path, images_path, saving_path):

	"""
    Function for permanent augumentation of trainingsset.
    Additionally it changes the Filetype to a more conv. PNG-File
    Assumption: Folders named '0' to '42' are existent under saving:path
    if thats not the case pls use something like:

     if not os.path.exists(myPath):
		 	os.makedirs(myPath)

	#Example:
    #saving_path = r'C:\...\Desktop\GTSRB_AugTrngset'

    """

    trng_images, trng_labels = loadGTSRBdataset(csv_path, images_path)

	enh_images = []
	enh_labels = []
	for image in trng_images:
		enh_images += augment_image(image, brightness=0.75, angle=12, translation=6, shear=2.5)

	for label in trng_labels:
		for i in range(8):
			enh_labels.append(label)

	for index in range(len(enh_images)):
		curFolderPath = saving_path + '/' + str(enh_labels[index])
		curFileCnt = next(os.walk(curFolderPath))[2] 
		cv2.imwrite(curFolderPath + '/GTSRB_' + str(enh_labels[index]) + '_' + str(len(curFileCnt)) + '.png' , cv2.resize(enh_images[index], (48,48)))

	with open(saving_path + '/' + 'AugTrng_Labels.csv',"w") as f:
	    wr = csv.writer(f,delimiter="\n")
	    wr.writerow(enh_labels)

		
def splitTrainingset(aug_images, aug_labels):
	"""
	Splits augumented trainingset into an equalized set with 200 images per class
	and a therefore reduced set with original distribution of images over all classes.
	
	#Note: You can combine this with the saving function above
	#Note: aug_images and aug_labels remain but are shortend -> print(len(aug_images))

    	"""
	eq_images = []
	eq_labels = []

	lower_bound = 0
	upper_bound = len(aug_images)

	#assumtion: class indizes are sorted and increasing
	for classindex in range(43):
		for position in range(lower_bound, upper_bound):
			if aug_labels[position] != classindex:
				upper_bound	= position
				break
		sample_index= random.sample(range(lower_bound, upper_bound), 200)  

		for i in sample_index:
			eq_images.append(aug_images[i][:])

		for i in sample_index:
			eq_labels.append(aug_labels[i])

		cnt = 0
		for i in sample_index:
			aug_images.pop(i - cnt)
			aug_labels.pop(i - cnt)
			cnt +=1

			lower_bound = upper_bound
		upper_bound = len(aug_images)

		

def readTrafficSigns(rootpath):
	"""
	Loading function for augumented trng-set images with new folder structure

    """
	images = [] 
	for c in range(0,43):
		curFileCnt = next(os.walk(rootpath +'/' + str(c)))[2] 
		for cnt in range(len(curFileCnt)):	
			curFilePath = rootpath + '/' + str(c) + '/GTSRB_' + str(c) + '_' + str(cnt) + '.png'
			images.append(cv2.cvtColor(cv2.imread(curFilePath), cv2.COLOR_BGR2RGB))
	return images


def readLabels(pathToCSV):
	"""
	Loading function for augumented trng-set labels with new folder structure

    """
	labels = [] 
	gtFile = open(pathToCSV) 
	gtReader = csv.reader(gtFile, delimiter="\n") 
	for row in gtReader:
		labels += row
	gtFile.close()
	return list(map(int, labels))



def random_brightness(image, ratio):
	"""
    Add random brightness. ratio should be around
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.float64(hsv[:, :, 2])
    brightness = brightness * (1.0 + np.random.uniform(-ratio, ratio))
    brightness[brightness>255] = 255
    brightness[brightness<0] = 0
    hsv[:, :, 2] = brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_rotation(image, angle):
	"""
    Add random rotation. angle should be around
    """

    if angle == 0:
        return image
    angle = np.random.uniform(-angle, angle)
    rows, cols = image.shape[:2]
    size = cols, rows
    center = cols/2, rows/2
    scale = 1.0
    rotation = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation, size)


def random_translation(image, translation):
	"""
    Add random translation. tranlation should be around
    """
    if translation == 0:
        return 0
    rows, cols = image.shape[:2]
    size = cols, rows
    x = np.random.uniform(-translation, translation)
    y = np.random.uniform(-translation, translation)
    trans = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image, trans, size)


def random_shear(image, shear):
	"""
    Add random shear. shear should be around
    """
    if shear == 0:
        return image
    rows, cols = image.shape[:2]
    size = cols, rows
    left, right, top, bottom = shear, cols - shear, shear, rows - shear
    dx = np.random.uniform(-shear, shear)
    dy = np.random.uniform(-shear, shear)
    p1 = np.float32([[left   , top],[right   , top   ],[left, bottom]])
    p2 = np.float32([[left+dx, top],[right+dx, top+dy],[left, bottom+dy]])
    move = cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image, move, size)
    
    
def augment_image(image, brightness, angle, translation, shear):
	"""
    Wrapper for augumention functions.
    """
	aug_images = []
	aug_images.append(image)
	aug_images.append(random_brightness(image, brightness))
	aug_images.append(random_rotation(image, angle))
	aug_images.append(random_translation(image, translation))
	aug_images.append(random_shear(image, shear))
	aug_images.append(random_rotation(random_brightness(image, brightness), angle))
	aug_images.append(random_translation(random_brightness(image, brightness), translation))
	aug_images.append(random_shear(random_brightness(image, brightness), shear))

	return aug_images




