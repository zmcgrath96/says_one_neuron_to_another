import sys, os
from os import walk
import numpy as np
from square_and_pickle_pics import *
from conv.CNN import CNN
from random import shuffle

# Params for training on a data set
training_folder = 'training_images/'
pickled_pics = {'simpsons': training_folder + '/simpsons/pickled_images/', 'dogs': training_folder + 'dogs/pickled_images/'}
images = {'simpsons': training_folder + 'simpsons/simpsons_dataset/', 'dogs': training_folder + 'dogs/images/'}
annotations = {'simpsons': training_folder + 'simpsons/annotation.txt', 'dogs': training_folder + 'annotation/'}
ouptut_folder = 'trained/'
img_size = 250
color = [0, 0, 0]


def train(s):
	if not (os.path.isdir(pickled_pics[s])):
		imgs = []
		for (path, _, filenames) in walk(images[s]):
			complete_paths = [path + '/' + f for f in filenames if os.path.isfile(path + '/' + f)]
			imgs.extend(complete_paths)

		num_output = len(os.listdir(images[s]))
		out_f = ouptut_folder + s + '.pickle'
		pickled_path = pickled_pics[s]
		np_pic_list = []

		if len(os.listdir(pickled_path)) == 0:
			print('Normalizing photos to {}px square images...'.format(img_size))
			for i in imgs:
				np_pic_list.append(square(i, img_size, color))
			print('Done\nPickling images...')
			pickle_pics(np_pic_list, pickled_path)
			print('Done')

	else:

		label_map = dict()
		index = 0
		num_img = 0
		for (path, _, filenames) in walk(pickled_pics[s]):
			num_img = len(filenames)
			for file in filenames:
				key = file.split(".")[0]
				if ("-" in key):
					key = key.split("-")[1]
				label_map[key] = index
				index += 1
		data_labels = []
		for (path, _, filenames) in walk(pickled_pics[s]):
			for img, i in zip(filenames, range(len(filenames))):
				key = img.split(".")[0]
				if ("-" in key):
					key = key.split("-")[1]
				data = np.load(path + "/" + img)
				label = label_map[key]
				data_labels.append([data, label])

		shuffle(data_labels)
		data = []
		labels = []
		for d, l in data_labels:
			data.append(d)
			label = np.zeros(len(label_map))
			label[l] = 1
			labels.append(label)
		data = np.array(data)
		labels = np.array(labels)
		cnn = CNN()
		cnn.train(data, img_size, labels, batch_size=2)




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