{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AUTOENCODERS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook you will find an explanation of what is an autoencoder, how it works, and see an implementation of an autoencoder in TensorFlow. By the end of this notebook, you should be able to create simple autoencoders and how to apply them to problems that involves unsupervised learning.\n",
    "\n",
    "<br>\n",
    "<br>\n",
    "<div class=\"alert alert-block alert-info\">\n",
    "<h2>Table of Contents</h2>\n",
    "<ol>\n",
    " <li><a href=\"#ref1\">Introduction</a></li>\n",
    " <li><a href=\"#ref2\">Feature Extraction and Dimensionality Reduction</a></li>\n",
    " <li><a href=\"#ref3\">Autoencoder Structure</a></li>\n",
    " <li><a href=\"#ref4\">Performance</a></li>\n",
    " <li><a href=\"#ref5\">Training: Loss Function</a></li>\n",
    " <li><a href=\"#ref6\">Code</a></li>\n",
    "</ol>\n",
    "</div>\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref1\"></a>\n",
    "\n",
    "## Introduction\n",
    "\n",
    "An autoencoder, also known as autoassociator or Diabolo networks, is an artificial neural network employed to recreate the given input.\n",
    "It takes a set of <b>unlabeled</b> inputs, encodes them and then tries to extract the most valuable information from them.\n",
    "They are used for feature extraction, learning generative models of data, dimensionality reduction and can be used for compression. \n",
    "\n",
    "A 2006 paper named <b><a href=\"https://www.cs.toronto.edu/~hinton/science.pdf\">Reducing the Dimensionality of Data with Neural Networks</a>, done by G. E. Hinton and R. R. Salakhutdinov</b>, showed better results than years of refining other types of network, and was a breakthrough in the field of Neural Networks, a field that was \"stagnant\" for 10 years.\n",
    "\n",
    "Now, autoencoders, based on Restricted Boltzmann Machines, are employed in some of the largest deep learning applications. They are the building blocks of Deep Belief Networks (DBN).\n",
    "\n",
    "<img src=\"https://ibm.box.com/shared/static/xlkv9v7xzxhjww681dq3h1pydxcm4ktp.png\" style=\"width: 350px; align: center;\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref2\"></a>\n",
    "\n",
    "## Feature Extraction and Dimensionality Reduction\n",
    "\n",
    "An example given by Nikhil Buduma in KdNuggets (<a href=\"http://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html\">link</a>) which gave an excellent explanation of the utility of this type of Neural Network.\n",
    "\n",
    "Say that you want to extract what emotion the person in a photography is feeling. Using the following 256x256 pixel grayscale picture as an example:\n",
    "\n",
    "<img src=\"https://ibm.box.com/shared/static/r5knpow4bk2farlvxia71e9jp2f2u126.png\">\n",
    "\n",
    "But when use this picture we start running into a bottleneck! Because this image being 256x256 pixels in size correspond with an input vector of 65536 dimensions! If we used an image produced with conventional cellphone cameras, that generates images of 4000 x 3000 pixels, we would have 12 million dimensions to analyze.\n",
    "\n",
    "\n",
    "This bottleneck is further problematized as the difficulty of a machine learning problem is increased as more dimensions are involved. According to a 1982 study by C.J. Stone (<a href=\"http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf\">link</a>), the time to fit a model, is optimal if:\n",
    "\n",
    "\n",
    "\n",
    "<div class=\"alert alert-block alert-info\" style=\"margin-top: 20px\">\n",
    "    \n",
    "### **$$m^{-p/(2p+d)}$$**\n",
    "\n",
    "Where:\n",
    "\n",
    "m: Number of data points\n",
    "\n",
    "d: Dimensionality of the data\n",
    "\n",
    "p: Parameter that depends on the model\n",
    "</div>\n",
    "\n",
    "As you can see, it increases exponentially!\n",
    "Returning to our example, we don't need to use all of the 65,536 dimensions to classify an emotion. A human identify emotions according to some specific facial expression, some <b>key features</b>, like the shape of the mouth and eyebrows.\n",
    "\n",
    "<img src=\"https://ibm.box.com/shared/static/m8urvuqujkt2vt1ru1fnslzh24pv7hn4.png\" height=\"256\" width=\"256\" align=\"center\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref3\"></a>\n",
    "\n",
    "## Autoencoder Structure\n",
    "\n",
    "<img src=\"https://ibm.box.com/shared/static/no7omt2jhqvv7uuls7ihnzikyl9ysnfp.png\" style=\"width: 400px;\">\n",
    "\n",
    "An autoencoder can be divided in two parts, the <b>encoder</b> and the <b>decoder</b>.\n",
    "\n",
    "The encoder needs to compress the representation of an input. In this case we are going to reduce the dimension the face of our actor, from 2000 dimensions to only 30 dimensions, by running the data through layers of our encoder.\n",
    "\n",
    "The decoder works like encoder network in reverse. It works to recreate the input, as closely as possible. This plays an important role during training, because it forces the autoencoder to select the most important features in the compressed representation.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref4\"></a>\n",
    "\n",
    "## Performance\n",
    "\n",
    "After the training has been done, you can use the encoded data as a reliable dimensionally-reduced data, applying it to any problems where dimensionality reduction seems appropriate.\n",
    "\n",
    "<img src=\"https://ibm.box.com/shared/static/yt3xyon4g2jyw1w9qup1mvx7cgh28l64.png\">\n",
    "\n",
    "This image was extracted from the G. E. Hinton and R. R. Salakhutdinovcomparing's <a href=\"https://www.cs.toronto.edu/~hinton/science.pdf\">paper</a>, on the two-dimensional reduction for 500 digits of the MNIST, with PCA on the left and autoencoder on the right. We can see that the autoencoder provided us with a better separation of data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref5\"></a>\n",
    "\n",
    "## Training: Loss function\n",
    "\n",
    "An autoencoder uses the Loss function to properly train the network. The Loss function will calculate the differences between our output and the expected results. After that, we can minimize this error with gradient descent. There are more than one type of Loss function, it depends on the type of data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Binary Values:\n",
    "$$l(f(x)) = - \\sum_{k} (x_k log(\\hat{x}_k) + (1 - x_k) \\log (1 - \\hat{x}_k) \\ )$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For binary values, we can use an equation based on the sum of Bernoulli's cross-entropy. \n",
    "\n",
    "$x_k$ is one of our inputs and $\\hat{x}_k$ is the respective output.\n",
    "\n",
    "We use this function so that if $x_k$ equals to one, we want to push $\\hat{x}_k$ as close as possible to one. The same if $x_k$ equals to zero.\n",
    "\n",
    "If the value is one, we just need to calculate the first part of the formula, that is, $- x_k log(\\hat{x}_k)$. Which, turns out to just calculate $- log(\\hat{x}_k)$.\n",
    "\n",
    "And if the value is zero, we need to calculate just the second part, $(1 - x_k) \\log (1 - \\hat{x}_k) \\ )$ - which turns out to be $log (1 - \\hat{x}_k) $.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Real values:\n",
    "$$l(f(x)) = - \\frac{1}{2}\\sum_{k} (\\hat{x}_k- x_k \\ )^2$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As the above function would behave badly with inputs that are not 0 or 1, we can use the sum of squared differences for our Loss function. If you use this loss function, it's necessary that you use a linear activation function for the output layer.\n",
    "\n",
    "As it was with the above example, $x_k$ is one of our inputs and $\\hat{x}_k$ is the respective output, and we want to make our output as similar as possible to our input."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Loss Gradient:\n",
    "\n",
    "$$\\nabla_{\\hat{a}(x^{(t)})} \\ l( \\ f(x^{(t)}))  = \\hat{x}^{(t)} - x^{(t)} $$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the gradient descent to reach the local minimum of our function $l( \\ f(x^{(t)})$, taking steps towards the negative of the gradient of the function in the current point.\n",
    "\n",
    "Our function about the gradient $(\\nabla_{\\hat{a}(x^{(t)})})$ of the loss of $l( \\ f(x^{(t)})$ in the preactivation of the output layer.\n",
    "\n",
    "It's actually a simple formula, it is done by calculating the difference between our output $\\hat{x}^{(t)}$ and our input $x^{(t)}$.\n",
    "\n",
    "Then our network backpropagates our gradient $\\nabla_{\\hat{a}(x^{(t)})} \\ l( \\ f(x^{(t)}))$ through the network using <b>backpropagation</b>."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"ref6\"></a>\n",
    "## Code\n",
    "\n",
    "For this part, we walk through a lot of Python 2.7.11 code. We are going to use the MNIST dataset for our example.\n",
    "The following code was created by Aymeric Damien. You can find some of his code in <a href=\"https://github.com/aymericdamien\">here</a>. We made some modifications for us to import the datasets to Jupyter Notebooks."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's call our imports and make the MNIST data available to use."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:521: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:522: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-1-aeda475fcce4>:10: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n",
      "WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please write your own downloading logic.\n",
      "WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting /tmp/data/train-images-idx3-ubyte.gz\n",
      "WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting /tmp/data/train-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.one_hot on tensors.\n",
      "Extracting /tmp/data/t10k-images-idx3-ubyte.gz\n",
      "Extracting /tmp/data/t10k-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n"
     ]
    }
   ],
   "source": [
    "#from __future__ import division, print_function, absolute_import\n",
    "\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "# Import MINST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, let's give the parameters that are going to be used by our NN."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rate = 0.01\n",
    "training_epochs = 20\n",
    "batch_size = 256\n",
    "display_step = 1\n",
    "examples_to_show = 10\n",
    "\n",
    "# Network Parameters\n",
    "n_hidden_1 = 256 # 1st layer num features\n",
    "n_hidden_2 = 128 # 2nd layer num features\n",
    "n_input = 784 # MNIST data input (img shape: 28*28)\n",
    "\n",
    "# tf Graph input (only pictures)\n",
    "X = tf.placeholder(\"float\", [None, n_input])\n",
    "\n",
    "weights = {\n",
    "    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),\n",
    "    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),\n",
    "    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),\n",
    "    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),\n",
    "}\n",
    "biases = {\n",
    "    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),\n",
    "    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),\n",
    "    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),\n",
    "    'decoder_b2': tf.Variable(tf.random_normal([n_input])),\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we need to create our encoder. For this, we are going to use sigmoidal functions. Sigmoidal functions delivers great results with this type of network. This is due to having a good derivative that is well-suited to backpropagation. We can create our encoder using the sigmoidal function like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Building the encoder\n",
    "def encoder(x):\n",
    "    # Encoder first layer with sigmoid activation #1\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))\n",
    "    # Encoder second layer with sigmoid activation #2\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))\n",
    "    return layer_2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And the decoder:\n",
    "\n",
    "You can see that the layer_1 in the encoder is the layer_2 in the decoder and vice-versa."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Building the decoder\n",
    "def decoder(x):\n",
    "    # Decoder first layer with sigmoid activation #1\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))\n",
    "    # Decoder second layer with sigmoid activation #2\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))\n",
    "    return layer_2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's construct our model.\n",
    "In the variable <code>cost</code> we have the loss function and in the <code>optimizer</code> variable we have our gradient used for backpropagation."
   ]
















import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train[0].shape
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
inputs =Input(shape=(784,))
enc = Dense(32,activation='relu') #compressing using 32 neurons
encoded = enc(inputs)
dec = Dense(784,activation='sigmoid')# decompress to 784 pixel
decoded = dec(encoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
def preprocess(x):
    x = x.astype('float32') / 255.
    return x.reshape(-1, np.prod(x.shape[1:])) # flatten

X_train = preprocess(X_train)
X_test  = preprocess(X_test)

#  a validation set for training
X_train, X_valid = train_test_split(X_train, test_size=500)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_valid, X_valid))
encoder = Model(inputs, encoded)
X_test_encoded = encoder.predict(X_test)
X_test_encoded[0].shape
decoder_inputs = Input(shape=(32,))
decoder = Model(decoder_inputs, dec(decoder_inputs))
X_test_decoded = decoder.predict(X_test_encoded)
def show_images(before_images, after_images):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        # before
        plt.subplot(2, 10, i+1)
        plt.imshow(before_images[i].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # after
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(after_images[i].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
show_images(X_test, X_test_decoded)
