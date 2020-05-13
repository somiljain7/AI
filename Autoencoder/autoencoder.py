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