# Says One Neuron to Another
Repository for EECS 738 Project Neural Network

#### Authors
Zachary McGrath, Kevin Ray (github.com/kray10)

### Data Sets
The number recognition dataset is from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_train.csv and the letters are from https://www.kaggle.com/crawford/emnist#emnist-letters-train.csv.

## Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/says_one_neuron_to_another
```

Install dependencies
```
$> pip3 install -r requirements.txt
```

For prediction, the number recogintion has been trained on the training set to a 92% accuracy and the letter recognizer to a 60% accuracy. The necessary weights have been provided in the trained folder.The prediciton can be run with:
```
%> python3 predict_nn.py -numbers # for testing number recogition
%> python3 predict_nn.py -letters # for testing letter recogition
```
In either case, a random sample will be taken from the test dataset (which has been provided) and a prediction will be made of which class it belongs to.

For retraining, the necessary trainging datasets will need to be download from the links above and the .csv will need to be placed in their corresponding training_images/ folders. The neural networks can then be trained with:
```
%> ptyhon3 train_nn.py -letters
or
%> ptyhon3 train_nn.py -numbers
```
The training command also has the following optional parameters:
```
-n=<number of nodes in hidden layer>
-r    # resume training using previously saved weights
-it=<max number of training iterations>
-a=<target accuracy to train to>
```

## Theory
### Network Choice


### Architecture

## Running
## Process
