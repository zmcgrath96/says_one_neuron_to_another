import sys, os 
from os import walk
import numpy as np
from square_and_pickle_pics import *

# Params for training on a data set
training_folder = 'training_images/'
pickled_pics = {'simpsons': training_folder + 'simpsons/pickled_images/', 'dogs': training_folder + 'dogs/pickled_images/'}
images = {'simpsons': training_folder + 'simpsons/simpsons_dataset/', 'dogs': training_folder + 'dogs/images/'}
annotations = {'simpsons': training_folder + 'simpsons/annotation.txt', 'dogs': training_folder + 'annotation/'}
ouptut_folder = 'trained/'
img_size = 250
color = [0, 0, 0]


def train(s):
	imgs = []
	for (path, _, filenames) in walk(images[s]):
		complete_paths = [path + '/' + f for f in filenames if os.path.isfile(path + '/' + f)]
		imgs.extend(complete_paths)

	num_output = len(os.listdir(images[s]))
	out_f = ouptut_folder + s + '.pickle'
	pickled_path = pickled_pics[s]
	np_pic_list = []
	
	if not os.path.isdir(pickled_path):
		print('Normalizing photos to {}px square images...'.format(img_size))
		for i in imgs:
			np_pic_list.append(square(i, img_size, color))
		print('Done\nPickling images...')
		pickle_pics(np_pic_list, pickled_path)
		print('Done')

	else:
		print('TODO: LOAD IN PICKLED IMAGES FOR TRAINING')


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