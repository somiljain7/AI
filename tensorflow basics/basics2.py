import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SimpleNeuralNetwork(Model):
  def __init__(self):
    super(SimpleNeuralNetwork , self).__init__()
    self.layer1 = Dense(2,activation='relu')
    self.layer2 = Dense(3,activation='relu')
    self.outputLayer = Dense(1,activation='softmax')
  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return self.outputLayer(x)

Model = SimpleNeuralNetwork()
     
 
