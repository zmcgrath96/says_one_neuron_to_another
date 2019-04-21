import sys
import cnn
import numpy as np
from os import walk
from random import shuffle

trained = "trained/"
training_folder = 'training_images/'
source = {"-m": "mnist_params.npy", "-s": "simpsons_params.npy"}
pickled_pics = {'simpsons': training_folder + 'simpsons/pickled_images/', 'dogs': training_folder + 'dogs/pickled_images/'}

def main(args):

    if '-m' in args[0]:
        print("Loading mnist test dataset...")
        params = np.load(trained + source[args[0]])
        dataset = np.genfromtxt("training_images/mnist/mnist_test.csv", skip_header=1, delimiter=",")
        data = dataset[:,1:].reshape((dataset.shape[0], 28, 28, 1))
        labels = dataset[:,0]
        print("Begining cnn testing...")
        cnn.test(data, labels, params)
    elif '-s' in args[0]:
        print("Loading simpsons test dataset...")
        params = np.load(trained + source[args[0]])
        data, labels, num_classes = load_images("simpsons")
        print("Begining cnn testing...")
        cnn.test(data, labels, params)

def load_images(s):
    label_map = dict()
    index = 0
    num_img = 0
    print("loading pickled files...")
    for (path, _, filenames) in walk(pickled_pics[s]):
    	if len(filenames) > 0:
    		key = path.split("/")[3]
    		label_map[key] = index
    		index += 1

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
    for d, l in data_labels:
    	data.append(d)
    	labels.append(l)
    data = np.array(data)
    labels = np.array(labels)
    data = data[0:100,:]
    labels = labels[0:100]
    return data, labels, len(label_map)




if __name__ == '__main__':
	main(sys.argv[1:])