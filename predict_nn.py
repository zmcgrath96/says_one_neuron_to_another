import numpy as np
import sys, os
from nn import NeuralNetwork
from train_nn import load_from_pickle
from train_nn import build_dataset
from train_nn import load_from_csv


def main(args):
    if args[0] == '-numbers':
        predict('mnist', '_', 10, args)
    elif args[0] == '-letters':
        predict('emnist-letters', '-', 26, args)

def predict(set, sep, num_classes, args):
    exist_test = os.path.isfile("training_images/" + set + "/" + set + sep +"test.pkl")
    if exist_test:
        dataset = load_from_pickle(set, sep + "test")
    else:
        dataset = load_from_csv(set, sep + "test")
    data, labels = build_dataset(dataset, num_classes)
    nn = NeuralNetwork()
    params = np.load("trained/" + set + "_params.npy")
    if '-a' in args:
        acc = nn.test(data, labels, params)
        print("Testing completed with {}% accuracy".format(round(acc*100,4)))
    else:
        ans = np.argmax(labels[0])
        guess = np.argmax(nn.predict(params, data[0]))
        print("Prediction was number {}, actual number was {}".format(guess, ans))

if __name__ == '__main__':
	main(sys.argv[1:])