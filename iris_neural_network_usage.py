# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:45:28 2020

@author: Sabato
"""

import keras
import numpy

# Loading Keras model
mnist_neural_network_json_file = open("iris_neural_network_model.json")
mnist_neural_network_json = mnist_neural_network_json_file.read()
mnist_neural_network_json_file.close()
neural_network = keras.models.model_from_json(mnist_neural_network_json)

# Loading weights for Keras model
neural_network.load_weights("iris_neural_network_model.h5")

neural_network.compile(loss='categorical_crossentropy', 
                       optimizer='adam', 
                       metrics=['accuracy'])

# Asking the features of the flower
first_parameter = input("Insert the first parameter (sepal length in cm):\n")
second_parameter = input("Insert the second parameter (sepal width in cm):\n")
third_parameter = input("Insert the third parameter (petal length in cm):\n")
fourth_parameter = input("Insert the fourth parameter (petal width in cm):\n")

# Input for the model
example = numpy.array((float(first_parameter),
                   float(second_parameter),
                   float(third_parameter),
                   float(fourth_parameter)))

# Prediction of the type of Iris
result = neural_network.predict(example.reshape(-1,4))
class_predicted = neural_network.predict_classes(example.reshape(-1,4))[0]

print("\nPrediction(%):\t", result[0])

if   (class_predicted==0):
    iris_type = "Iris Setosa"
elif (class_predicted==1):
    iris_type = "Iris Versicolour"
elif (class_predicted==2):    
    iris_type = "Iris Virginica"
    
print("Flower:\t\t", iris_type)
