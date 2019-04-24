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
## Running
For prediction, the number recogintion has been trained on the training set to a 92% accuracy and the letter recognizer to a 58% accuracy. The necessary weights have been provided in the trained folder.The prediciton can be run with:
```
%> python3 predict_nn.py -numbers     # for testing number recogition
%> python3 predict_nn.py -letters     # for testing letter recogition
```
In either case, a random sample will be taken from the test dataset (which has been provided) and a prediction will be made of which class it belongs to.

To test predictions on the whole training set, the "-a" command can be added to the command line. This will test the accuracy of all elements in the test dataset and return the percent that were predicted correctly.

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
Neural networks (NN) and all of their various forms have many different applications. Different architectures are good for different data sets. For example, convolutional neural networks are good for image regocnititon as they find defining features of images. The applications for neural networks are numerous and cannot possibly be enumerated here. 

The theory behind NN is based in how the human brain works. 'Nodes' take a certain input, perform some sort of work function and an applied bias, and produces some output. By constantly changing the values of these weights and biases based on the error of the forward propagation, a more accurate NN can be trained.

Training the network itself can be fairly tricky. Trying to make an accurate NN without overfitting can be tough. Training in itself has many variables, such as the number of times to run through the data set (or Epochs), the size of each batch, and learning rate. All of these are major contributors to how well the NN performs. 
## Network and Architecture
The Network choice here was a simple fully connected network, consisting of 784 input nodes, 1 hidden layer of 100 nodes, and an output layer of 10 nodes for the number set and 26 for the letter set. 784 was chosen as the number of input nodes as each image in the dataset is 28 x 28 pixels, or a total of 784. The simple fully connected network was chosen due to its ease of implementation while also being able to gain a firm understanding of the mechanisms at play, namely forward and backward propagation.
The work function for this network is the sigmoid function. 

## Results
As stated above, the MNIST integer data set yielded an accuracy of 92%, and the EMNIST letter data set an accuracy of 58%. The 92% accuracy of the integer data set is satisfactory seeing as looking at some of the numbers ourselves as people, we had issues determining what some of the numbers were. While 58% accuracy of the letter data set is far from perfect, it is well above the accuracy that guessing would yeild which would be 1/26, or about 4%.
## Future Work 
Future work for improving this simple NN could be adding in another layer between the input layer and the first hidden layer. As of now, the number of nodes decreases by a factor of almost 8 for the first hidden layer. By adding another layer with maybe 200 nodes, the accuracy of the letter set maight increase by a substantial percentage. 

## Things to note
The original plan for this project was to use a Convolutional Neural Network to classify two data sets: images of the Simpsons show, and dog breeds from Stanford. Unfortunately, accuracy never made its way beyond twice that of guessing. Two different attempts at training were made: one with images scaled down to 64x64 px and one with images scaled down to 128x128 px. 4 Epochs were used to train and the data was shuffled each Epoch. These attempts can be seen in the branches 64px-images and 128pxCNN. The training time on these took about a day, so after several attempts at each and not achieving a high accuracy, we decided to go for something more achieable. 