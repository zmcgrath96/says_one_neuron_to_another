import sys, os
from os import walk
import numpy as np
from square_and_pickle_pics import *
from random import shuffle
import cnn
import pickle

# Params for training on a data set
training_folder = 'training_images/'
pickled_pics = {'simpsons': training_folder + 'simpsons/pickled_images/', 'dogs': training_folder + 'dogs/pickled_images/'}
images = {'simpsons': training_folder + 'simpsons/simpsons_dataset/', 'dogs': training_folder + 'dogs/images/'}
annotations = {'simpsons': training_folder + 'simpsons/annotation.txt', 'dogs': training_folder + 'annotation/'}
ouptut_folder = 'trained/'
img_size = 64
color = [0, 0, 0]

np.set_printoptions(threshold=sys.maxsize)


def predict(s, path):
	if not os.path.isfile(path):
		raise Exception('File path is not a valid file')

	label_map = None
	with open(ouptut_folder + s + "_lable_dict.pickle", 'rb') as r:
		label_map  = pickle.load(r)

	# based on the mode, load in the np params
	params = np.load(ouptut_folder + s + "_params.npy")
	_, img = square(path, img_size, color)
	res = cnn.predict(img, params)
	res_arg = np.argmax(res)
	finding = ''
	for key in label_map:
		if label_map[key] == res_arg:
			finding = key
			break

	print('The image is of {}'.format(finding))

def accuracy(s):
	label_map = None
	with open(ouptut_folder + s + "_lable_dict.pickle", 'rb') as r:
		label_map  = pickle.load(r)

	data_labels = []
	for (path, _, filenames) in walk(pickled_pics[s]):
		if len(filenames) > 0:
			key = path.split("/")[3]
			print("Loading {} images...".format(key))
			for img, i in zip(filenames, range(len(filenames))):
				data = np.load(path + "/" + img)
				label = label_map[key]
				data_labels.append([data, label])

	print('Finished laoding images')
	params = np.load(ouptut_folder + s + "_params.npy")

	print('Testing accuracy...')
	right = count = 0
	num_imgs = len(data_labels)
	for d, l in data_labels:
		label = np.zeros(len(label_map))
		label[l] = 1
		out = cnn.predict(d, params)
		right = right + 1 if np.argmax(out) == np.argmax(label) else right 
		count += 1
		sys.stdout.write('Total progress: %d%%   \r' % ((count / num_imgs) * 100) )
		sys.stdout.flush()

	print('Model is about {}% correct'.format(int((right / count) * 100)))


def train(s):
	if not (os.path.isdir(pickled_pics[s])):
		imgs = []
		for (path, _, filenames) in walk(images[s]):
			complete_paths = [path + '/' + f for f in filenames if os.path.isfile(path + '/' + f)]
			imgs.extend(complete_paths)

		num_output = len(os.listdir(images[s]))
		out_f = ouptut_folder + s + '.pickle'
		pickled_path = pickled_pics[s]

		if not os.path.isdir(pickled_path):
			np_pic_list = []
			count = 1
			print('Normalizing photos to {}px square images...'.format(img_size))
			for i in imgs:
				np_pic_list.append(square(i, img_size, color))
				if count % 1000 == 0 or len(imgs) - count < 1000:
					print('Pickling images {} - {}...'.format(count - 1000, count))
					pickle_pics(np_pic_list, pickled_path)
					np_pic_list.clear()
				count += 1
		train_cnn(s)

	else:
		train_cnn(s)


def train_cnn(s):
	label_map = dict()
	index = 0
	print("loading pickled files...")
	for (path, _, filenames) in walk(pickled_pics[s]):
		if len(filenames) > 0:
			key = path.split("/")[3]
			label_map[key] = index
			index += 1

	with open(ouptut_folder + s + "_lable_dict.pickle", 'wb') as f:
		pickle.dump(label_map, f, protocol=pickle.HIGHEST_PROTOCOL)

	data_labels = []
	for (path, _, filenames) in walk(pickled_pics[s]):
		if len(filenames) > 0:
			key = path.split("/")[3]
			print("Loading {} images...".format(key))
			for img, i in zip(filenames, range(len(filenames))):
				data = np.load(path + "/" + img)
				label = label_map[key]
				data_labels.append([data, label])
				
	print('Finished loading images')
	shuffle(data_labels)
	data = []
	labels = []
	print('Loading labels...')
	for d, l in data_labels:
		data.append(d)
		label = np.zeros(len(label_map))
		label[l] = 1
		labels.append(label)
	print('Finished loading labels')
	data = np.array(data)
	labels = np.array(labels)
	params = cnn.train(data, labels, len(label_map), img_dim=img_size, batch_size=20, epochs=4)
	np.save(ouptut_folder + s + "_params", params)


def main(args):
	try: 
		if '-t' in args[0]:
			if '-s' in args[1]:
				print('Training on simpsons data set...')
				train('simpsons')

			elif '-d' in args[1]:
				print('Training on stanford dog data set...')
				train('dogs')

		elif '-p' in args[0]:
			if '-s' in args[1]:
				if '-a' in args[2]:
					accuracy('simpsons')
				else:
					predict('simpsons', str(args[2]))

			elif '-d' in args[1]:
				if '-a' in args[2]:
					accuracy('dogs')
				else:
					predict('dogs', str(args[2]))
	
		else:
			print('Invalid parameters: ')
			print('Correct call: python3 main.py <mode> <set> <img>')
			print('Training: <mode>: -t, <set>: -s (Simpsons), -d (dogs), <img>: none')
			print('Prediction: <mode>: -p, <set> -s (Simpsons), -d (dogs), <img, accuracy>: path to image or -a to test accuracy of model')

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main(sys.argv[1:])