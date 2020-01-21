import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os, random
import pickle
from findexer import *
		
i = IndexerTrain()
i.indexer()	


# class Indexer (object):
# 	def __init__(self):
# 		self.sift = cv.xfeatures2d.SIFT_create()
		
		
# 	def extract_features(self,image_path):
# 		print("Image path", image_path)
# 		img = cv.imread(image_path, 0)
# 		#img = cv.resize(img_i,(512, 384))
# 		# if img.ndim == 2:
# 		# 	gray_image = img
# 		# else:
# 		# 	gray_image = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

# 		kp, desc = self.sift.detectAndCompute(img, None)
# 		return desc
		
# 	def batch_extractor(self,images_path):
# 		files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
# 		result = {}
		
# 		for f in files:
# 			print('Extracting features from image {0}'.format(f))
# 			name = f.split('/')[-1].lower()
# 			#print("File Name", name)
# 			result[name] = self.extract_features(f)
# 			with open('image_db.pck', 'wb') as fp:
# 				pickle.dump(result, fp)
	
	
# indexer = Indexer()
# images_path = 'images/'
# indexer.batch_extractor(images_path)	
# 		