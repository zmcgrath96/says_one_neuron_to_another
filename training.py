import sys, os 
from os import walk
import numpy as np
import cv2
import pickle

# Params for training on a data set
training_folder = 'training_images/'
images = {'simpsons': training_folder + 'simpsons/simpsons_dataset/', 'dogs': training_folder + 'dogs/images/'}
annotations = {'simpsons': training_folder + 'simpsons/annotation.txt', 'dogs': training_folder + 'annotation/'}
ouptut_folder = 'trained/'
img_size = 250
color = [0, 0, 0]

def block_imgs(path):
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

		return (path ,cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color))
	
	except Exception as e:
		print('Could not open image: ' + str(path))


def train(s):
	imgs = []
	for (path, _, filenames) in walk(images[s]):
		complete_paths = [path + '/' + f for f in filenames if os.path.isfile(path + '/' + f)]
		imgs.extend(complete_paths)

	num_output = len(os.listdir(images[s]))
	out_f = ouptut_folder + s + '.pickle'

	sqr_imgs = []
	if 'simpsons' in s: 
		#cache the blocked images
		pickle_path = 'training_images/simpsons/pickled_images/'
		for i in imgs:
			if not os.path.isfile(i):
				continue
			sqr_imgs.append(block_imgs(i))

		print('pickling square images')
		
		for t in sqr_imgs:
			try:
				if t is None:
					continue
				path, data = t
				name = pickle_path + '/'.join(path.split('/')[-2:-1]).split('.')[0] + '.pickle'
				np.save(name, data)
			except Exception as e:
				print('failed with: ' + str(t))

	else:
		print('TODO: Annotations for dogs')


def main(args):
	if '-s' in args[0]:
		print('Training on simpsons data set...')
		train('simpsons')
	
	elif '-d' in args[0]:
		print('Training on stanford dog data set...')
		train('dogs')
	
	else:
		print('Invalid parameters: ')
		print('Correct call: python3 training.py <command>')
		print('command: -s for simpsons data set, -d for dog data set')



if __name__ == '__main__':
	main(sys.argv[1:])