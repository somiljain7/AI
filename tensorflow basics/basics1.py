import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=2 ,activation='relu'))
model.add(Dense(1, activation='softmax'))
 
