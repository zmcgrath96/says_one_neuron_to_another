import sys, os 
from os import walk
import numpy as np
import cv2

training_folder = 'training_images/'
images = {'simpsons': training_folder + 'simpsons/simpsons_dataset/', 'dogs': training_folder + 'dogs/images/'}
annotations = {'simpsons': training_folder + 'simpsons/annotation.txt', 'dogs': training_folder + 'annotation/'}
ouptut_folder = 'trained/'
img_size = 250
color = [0, 0, 0]

def block_imgs(path):
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

	return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def train(s):
	imgs = []
	for (path, _, filenames) in walk(images[s]):
		complete_paths = [path + '/' + f for f in filenames if os.path.isfile(path + '/' + f)]
		imgs.extend(complete_paths)

	num_output = len(os.listdir(images[s]))
	out_f = ouptut_folder + s + '.pickle'

	sqr_imgs = []
	if 'simpsons' in s: 
		for i in imgs:
			sqr_imgs.append(block_imgs(i))

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