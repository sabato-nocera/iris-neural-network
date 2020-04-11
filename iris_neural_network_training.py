# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:15:55 2020

@author: Sabato Nocera

Iris Plants Database: http://archive.ics.uci.edu/ml/datasets/Iris

Number of Instances: 150 (50 in each of three classes)

Number of Attributes: 4 numeric, predictive attributes and the class

Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
"""

import pandas
import numpy
from keras import models
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("iris.data", header=None,sep = ",")

iris_setosa = dataframe.values[0:50]
iris_versicolour = dataframe.values[50:100]
iris_virginica = dataframe.values[100:150]

features_iris_setosa = iris_setosa[ : , 0:4]
features_iris_versicolour = iris_versicolour[ : , 0:4]
features_iris_virginica = iris_virginica[ : , 0:4]

labels_iris_setosa = iris_setosa[ : , 4]
labels_iris_versicolour = iris_versicolour[ : , 4]
labels_iris_virginica = iris_virginica[ : , 4]

x_train = numpy.concatenate((features_iris_setosa[0:35], 
                             features_iris_versicolour[0:35], 
                             features_iris_virginica[0:35]), axis=0)
x_test = numpy.concatenate((features_iris_setosa[35:50], 
                            features_iris_versicolour[35:50], 
                            features_iris_virginica[35:50]), axis=0)
y_train = numpy.concatenate((labels_iris_setosa[0:35], 
                             labels_iris_versicolour[0:35], 
                             labels_iris_virginica[0:35]), axis=0)
y_test = numpy.concatenate((labels_iris_setosa[35:50], 
                            labels_iris_versicolour[35:50], 
                            labels_iris_virginica[35:50]), axis=0)

# LabelEncoder is used to convert the strings that represents the categories 
# into numbers
encoder = LabelEncoder()

encoder.fit(y_train)
y_train = encoder.transform(y_train)
# One Hot Encoded
y_train = to_categorical(y_train)

encoder.fit(y_test)
y_test = encoder.transform(y_test)
# One Hot Encoded
y_test = to_categorical(y_test)

neural_network = models.Sequential()
# Adding Dense layers, whose neurons receive input from the neurons of previous 
# layer.
#    1) The 1st parameter of a Dense layer represents the number of possible 
#        outputs that the layer produces (the last layer produces ten possible 
#                                         output because ten are the possible 
#                                         digits)
#    2) The activation function is used to calculate the output of each neuron 
#       of the layer (that output depends on its input)
#    3) "input_dim" is equal to "4" because four are the input features
neural_network.add(Dense(400, activation='tanh', input_dim=4))
neural_network.add(Dense(400, activation='relu'))
neural_network.add(Dense(3, activation='softmax'))

neural_network.compile(loss='categorical_crossentropy', 
                       optimizer='adam', 
                       metrics=['accuracy'])

neural_network.fit(x_train, y_train, epochs=5, batch_size=20)

test_loss, test_accuracy = neural_network.evaluate(x_test, y_test)

print('test_accuracy:', test_accuracy, 'test_loss', test_loss)

# Saving the model
json_model = neural_network.to_json()
with open("iris_neural_network_model.json", "w") as json_file:
    json_file.write(json_model)
neural_network.save_weights("iris_neural_network_model.h5")

# Making predictions
example = numpy.array((5.1,3.5,1.4,0.2))
print("\nExample:\t", example, " is a Iris Setosa (1st position)")
print("Prediction(%): \t", neural_network.predict(example.reshape(-1,4))[0])

class_predicted = neural_network.predict_classes(example.reshape(-1,4))[0]

if   (class_predicted==0):
    iris_type = "Iris Setosa"
elif (class_predicted==1):
    iris_type = "Iris Versicolour"
elif (class_predicted==2):    
    iris_type = "Iris Virginica"
    
print("Class predicted:", iris_type)
