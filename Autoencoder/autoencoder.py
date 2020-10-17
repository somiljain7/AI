## AUTOENCODERS

In this notebook you will find an explanation of what is an autoencoder, how it works, and see an implementation of an autoencoder in TensorFlow.
By the end of this notebook, you should be able to create simple autoencoders and how to apply them to problems that involves unsupervised learning.



## Introduction
An autoencoder, also known as autoassociator or Diabolo networks, is an artificial neural network employed to recreate the given input. It takes a set of unlabeled inputs, encodes them and then tries to extract the most valuable information from them. They are used for feature extraction, learning generative models of data, dimensionality reduction and can be used for compression.

A 2006 paper named Reducing the Dimensionality of Data with Neural Networks, done by G. E. Hinton and R. R. Salakhutdinov, showed better results than years of refining other types of network, and was a breakthrough in the field of Neural Networks, a field that was "stagnant" for 10 years.

Now, autoencoders, based on Restricted Boltzmann Machines, are employed in some of the largest deep learning applications. They are the building blocks of Deep Belief Networks (DBN).




## Autoencoder Structure


An autoencoder can be divided in two parts, the encoder and the decoder.

The encoder needs to compress the representation of an input. In this case we are going to reduce the dimension the face of our actor, from 2000 dimensions to only 30 dimensions, by running the data through layers of our encoder.

The decoder works like encoder network in reverse. It works to recreate the input, as closely as possible. This plays an important role during training, because it forces the autoencoder to select the most important features in the compressed representation.



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
