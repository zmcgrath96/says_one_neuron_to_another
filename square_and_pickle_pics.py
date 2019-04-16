import numpy as np 
import pickle
import cv2
import os

def square(path, img_size, color):
	try:
		img = cv2.imread(path)
		# size is in (height, width) format
		old_size = img.shape[:2] 

		ratio = float(img_size)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])

		# new_size should be in (width, height) format

		img = cv2.resize(img, (new_size[1], new_size[0]))

		delta_w = img_size - new_size[1]
		delta_h = img_size - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)

		path = '/'.join(path.split('/')[-2:])
		return (path ,cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color))
	
	except Exception as e:
		print('Could not open image: ' + str(path))

def pickle_pics(pic_list, pickle_path):
	if 'list' not in str(type(pic_list)):
		print('Invalid parameter: first parameter must be list of np arrays')
		return
	for t in pic_list:
		try:
			if t is None:
				continue
			path, data = t
			name = pickle_path + '/' + path.split('.')[0] + '.pickle'
			name = name.replace('//','/')
			directory =  '/'.join(name.split('/')[:-1])
			if not os.path.exists(directory):
				os.makedirs(directory)
			np.save(name, data)
		except Exception as e:
			print(e)