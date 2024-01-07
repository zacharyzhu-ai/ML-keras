from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) =  mnist.load_data()

image_index = 19482 # You may select anything up to 60,000
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
network.compile(optimizer = 'rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print (network.summary())

prob_test = (network.predict(test_images)).astype("float")
predict_test = np.argmax(prob_test, axis = -1)

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
#right = stacked_test_images[:,:][predict_test == before_categ_test_labels]
#wrong = stacked_test_images[:,:][predict_test != before_categ_test_labels]


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

correct_img = most_correct_guesses[:,3:].reshape((len(most_correct_guesses),28,28))
incorrect_img = most_incorrect_guesses[:,3:].reshape((len(most_incorrect_guesses),28,28))


fig1, ax1 = plt.subplots(3,3)
fig1.suptitle("worst incorrect predictions")
for i, ax1 in enumerate(ax1.flatten()):
    p = most_incorrect_guesses[i,0].astype(int).astype(str)
    a = most_incorrect_guesses[i,1].astype(int).astype(str)
    ax1.set_title('P: ' + p + 'A: ' + a)
    ax1.imshow(incorrect_img[i], cmap='gray_r')
plt.show()
fig2, ax2 = plt.subplots(3,3)
fig2.suptitle("best correct predictions")
for i, ax2 in enumerate(ax2.flatten()):
    p = most_correct_guesses[i,0].astype(int).astype(str)
    a = most_correct_guesses[i,1].astype(int).astype(str)
    ax2.set_title('P: ' + p + 'A: ' + a)
    ax2.imshow(correct_img[i], cmap='gray_r')
plt.show()