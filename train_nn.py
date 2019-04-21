import numpy as np
import sys, os
from nn import NeuralNetwork


def main(args):
    if args[0] == '-numbers':
        run_training('mnist', '_', args, 10)
    elif args[0] == '-letters':
        run_training('emnist-letters', '-', args, 26)

def run_training(set, sep, args, num_classes):
    exist_tr = os.path.isfile("training_images/" + set + "/" + set + sep + "train.pkl")
    exist_test = os.path.isfile("training_images/" + set + "/" + set + sep +"test.pkl")
    dataset_train = None
    dataset_test = None
    if exist_tr and exist_test:
        dataset_train = load_from_pickle(set, sep + "train")
        dataset_test = load_from_pickle(set, sep + "test")
    else:
        dataset_train = load_from_csv(set, sep + "train")
        dataset_test = load_from_csv(set, sep + "test")

    img_size = 28
    img_pixels = img_size * img_size

    print("Training neural network...")
    iter_limit = 100
    for i, arg in zip(range(len(args)), args):
        if '-it=' in arg:
            iter_limit = int(arg.split("=")[1])

    nodes = 100
    for i, arg in zip(range(len(args)), args):
        if '-n=' in arg:
            nodes = int(arg.split("=")[1])

    acc_limit = 0.85
    for i, arg in zip(range(len(args)), args):
        if '-a=' in arg:
            acc_limit = float(arg.split("=")[1])
            if acc_limit >= 1.0:
                print("Accuracy threshold is too high. Reset to 0.85...")
                acc_limit = 0.85

    nn = NeuralNetwork(img_pixels, num_classes, nodes)

    params = None
    if '-r' in args:
        print("Resuming from previous training...")
        params = np.load("trained/" + set + "_params.npy")
    acc = 0.0
    iter = 0
    while acc < acc_limit and iter < iter_limit:
        data_train, labels_train = build_dataset(dataset_train, num_classes)
        data_test, labels_test = build_dataset(dataset_test, num_classes)
        params = nn.train(data_train, labels_train, params)
        np.save("trained/" + set + "_params", params)
        acc = nn.test(data_test, labels_test)
        print("Iteration {} completed with {}% accuracy".format(iter, round(acc*100,4)))
        iter += 1

def load_from_pickle(set, type):
    print("Loading mnist " + set + " from pickle...")
    with open("training_images/" + set + "/" + set + type + ".pkl", "br") as fh:
        data= np.load(fh)
    return data

def load_from_csv(set, type):
    print("Loading " + set + " " + type + " from csv...")
    dataset = np.loadtxt("training_images/" + set + "/" + set + type + ".csv", skiprows=1, delimiter=",")
    print("Pickling data...")
    with open("training_images/" + set + "/"  + set + type + ".pkl", "bw") as fh:
        np.save(fh, dataset)
    return dataset

def build_dataset(dataset, num_classes):
    np.random.shuffle(dataset)
    data = dataset[:,1:]
    shape = data.shape
    data = data.reshape((shape[0], shape[1], 1))
    data = np.asfarray(data) * 0.99 / 255 + 0.01
    labels = np.zeros((len(dataset), num_classes, 1))
    for i in range(dataset.shape[0]):
        index = int(dataset[i,0])
        labels[i, index - 1, 0] = 1
    labels[labels==0] = 0.01
    labels[labels==1] = 0.99
    return data, labels

if __name__ == '__main__':
	main(sys.argv[1:])