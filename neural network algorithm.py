# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:06:36 2023

@author: 1055842
"""

from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) =mnist.load_data()
image_index = 7777 # You may select anything up to 60,000
print(train_labels[image_index]) # The label is 8
plt.imshow(train_images[image_index], cmap='Greys')
one_image = train_images[image_index]
print ('Train Shape', train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)
test_images = test_images.reshape((10000, 28*28))
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
one_image = one_image.astype('float32')/255
before_categ_test_labels = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc * 100, "%")
print (network.summary())


def select_worst_best(before_categ_test_labels, predict_test, prob_test, test_images):
    # =============================================================================
    # Create a numpy array of max probabilities (using prob)
    # =============================================================================
    max_prob = np.max(prob_test, axis=1)

    # =============================================================================
    # Stack the predictions, actuals and probabilities with images
    # =============================================================================
    stacked_test_images = np.hstack((before_categ_test_labels.reshape(-1,1),predict_test.reshape(-1,1), max_prob.reshape(-1,1),test_images))

    # =============================================================================
    # Choose wrong and correct predictions
    # =============================================================================
    right = stacked_test_images[stacked_test_images[:,0]==stacked_test_images[:,1]]
    wrong = stacked_test_images[stacked_test_images[:,0]!=stacked_test_images[:,1]]

    # =============================================================================
    # Sort by probabilities
    # =============================================================================
    right = right[right[:,2].argsort()]
    wrong = wrong[wrong[:,2].argsort()]

    # =============================================================================
    # Choose the 9 highest probabilities of both (yes I know there are only 2)
    # =============================================================================
    most_correct_guesses = right[:9]
    most_incorrect_guesses = wrong[-9:]

    return most_incorrect_guesses, most_correct_guesses


############################################################
before_categ_test_labels = np.array([3.,6.,1.,0.])

predict_test = np.array([3.,6.,4.,1.])

prob = np.array([
    [.1,.0,.1,.3,.05,.01,.01,.01,.01,.01], 
    [.1,.2,.1,.01,.05,.01,.5,.01,.01,.01],
    [.1,.2,.0,.05,.6,.01,.01,.01,.01,.01],
    [.0,.9,.05,.02,.01,.05,.01,.01,.01,.03]  
])

test_images = np.array([
    [0.3,0.0,0.0,.7,.6,.51,.0,.0,.0,.0], 
    [0.6,0.0,0.0,.7,.6,.51,.0,.0,.0,.0],
    [0.1,0.0,0.0,.7,.6,.51,.0,.0,.0,.0],
    [0.0,0.0,0.0,.7,.6,.51,.0,.0,.0,.0]  
])


most_incorrect_guesses, most_correct_guesses  =select_worst_best(before_categ_test_labels,predict_test,prob,test_images)

print(most_incorrect_guesses)
print(most_correct_guesses)