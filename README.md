# Traffic Sign Classifier
This project was part of the Udacity's Self Driving Car Nanodegree. In this project, I have built and trained a deep neural network to classify traffic signs, using TensorFlow, experimented with different network architectures, performed image pre-processing and validation to guard against overfitting.

## Dataset
For the purpose of this project, I have used the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The dataset is summarized as follows:

The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32,32,3)
The number of unique classes/labels in the data set is 43

![Dataset](/writeup/dataset.png "Dataset")

## Model Architecture
My final model consisted of the following layers:

| Layer        | Description    |
| ------------- |:-------------:|
|Input| 32x32x1 Grayscale image|
|Convolution 3x3 |1x1 stride, valid padding, outputs 28x28x6|
|RELU|
|Max pooling| 2x2 stride, outputs 14x14x6|
|Convolution 3x3 |1x1 stride, valid padding, outputs 10x10x16|
|RELU|
|Max pooling |2x2 stride, outputs 5x5x16|
|Fully connected| Input: 400(5x5x16) Output: 120|
|RELU|
|Dropout |Keep_probability = 0.5|
|Fully connected |Output: 84|
|RELU|
|Dropout |Keep_probability = 0.5|
|logits |Output: 43|

## Training
To train the model, I used the Adam Optimizer as it is the most robust optimizer and is
recommended in all discussion formus. As it uses momentum, it helps in reaching the
minimum faster and avoids deviation of the loss function. The batch size is taken as 256 and
number of epochs is 47. The learning rate is 0.001. The parameters were chosen based on
intuition and trial and error.

## Results
My final model had a set accuracy of about 95.5% and a test set accuracy of 90.1%. To
implement this, firstly, I directly used the LeNet architecture from the exercises, with a few
modifications such as the input channel had three features instead of one and the final layer
had 43 classes instead of 10. This gave me an accuracy of about 89%. The main problem
was that in the final layers of the architecture, a lot of neurons were overfitting. Thus to
tackle this problem, I introduced 2 dropout layers to after each activation of the fully
connected layers to prevent overfitting. Finally, the hyperparameters were tuned using
intuition and trial and error.

>For further details about the project refer to the project writeup.
