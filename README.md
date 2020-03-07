# Kaggle-Flower-Recognition-Using-CNN

This project uses a Convolutional Neural Network (CNN) to predict flowers of 5 types using the flower recognition dataset on Kaggle.
https://www.kaggle.com/alxmamaev/flowers-recognition 
There are 5 types of flowers that are predicted and trained on:

Daisy </br>
Dandelion</br>
Rose</br>
Sunflower</br>
Tulip</br>
There are 4242 images in the original dataset.

I use a Pretrained ResNet-50 convolutional neural network model to do training and predictions. ResNet-50 is a convolutional neural network that is trained on more than a million images from the ImageNet database . The network is 50 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.

![resnet50 architecture](https://www.researchgate.net/publication/331364877/figure/fig3/AS:741856270901252@1553883726825/Left-ResNet50-architecture-Blocks-with-dotted-line-represents-modules-that-might-be.png) 

categorical_crossentropy is used for loss and Adam is used as the optimizer. I use ReLu within my layers and softmax as the activation function. Within my CNN, I take advantage of avgpooling.


I have managed a 87% accuracy during training.
