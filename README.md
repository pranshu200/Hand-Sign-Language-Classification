# Hand-Sign-Language-Classification
Objective: The project aims at building a machine learning model that will be able to classify the various hand gestures used for fingerspelling in sign language. In this model, classification machine learning algorithms are trained using a set of image data.

# Dataset
The "ASL Alphabet Dataset" was taken from Kaggle and used to train the model.<br>
👉 Dataset Link : https://www.kaggle.com/grassknoted/asl-alphabet

# Language and Framework used
<ul>
  <li>Language : Python</li>
  <li>NumPy: library for numerical calculations</li>
  <li>Pandas: library for data manipulation and analysis</li>
  <li>Tensorflow: library for large numerical computations without keeping deep learning in mind</li>
  <li>Keras: neural network library </li>
  <li>Matplotlib: for creating static, animated, and interactive visualizations</li>
</ul>

# CNN
Convolutional Neural Networks (CNN), are deep neural networks used to process data that have a grid-like topology, e.g images that can be represented as a 2-D array of pixels. A CNN model consists of four main operations: Convolution, Non-Linearity (Relu), Pooling and Classification (Fully-connected layer ).<br>

Convolution: The purpose of convolution is to extract features from the input image. It preserves the spatial relationship between pixels by learning image features using small squares of input data. It is usually followed by Relu.<br>

Relu: It is an element-wise operation that replaces all negative pixel values in the feature map by zero. Its purpose is to introduce non-linearity in a convolution network.<br>

Pooling: Pooling (also called downsampling ) reduces the dimesionality of each feature map but retains important data.<br>

Fully-connected layer: It is a multi layer perceptron that uses softmax function in the output layer. Its purpose is to use features from previous layers for classsifying the input image into various classes based on training data.<br>

The combination of these layers is used to create a CNN model. The last layer is a fully connected layer.<br>


# HoG (Histogram of Gradients):

A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information.<br>

Hog is a feature descriptor that calculates a histogram of gradient for the image pixels, which is a vector of 9 bins (numbers ) corresponding to the angles: 0, 20, 40, 60... 160. The images are divided into cells, (usually, 8x8 ), and for each cell, gradient magnitude and gradient angle is calculated, using which a histogram is created for a cell. The histogram of a block of cells is normalized, and the final feature vector for the entire image is calculated.

# gTTS (Google Text To Speech):

Google Text-to-Speech is a screen reader application developed by Google for the Android operating system. It powers applications to read aloud the text on the screen with support for many languages.<br>
gTTS (Google Text-to-Speech)is a Python library and CLI tool to interface with Google Translate text-to-speech API, save function allows us to save the converted speech in a format that allows us to play sounds. It saves the file in .mp3 format.