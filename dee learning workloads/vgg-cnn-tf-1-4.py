# Kudos to Danial Khosraivy for his blog on using a VGG-like CNN to get good accuracy on the Fashion MNIST dataset. 
# http://danialk.github.io/blog/2017/09/29/range-of-convolutional-neural-networks-on-fashion-mnist-dataset/
# I use his VGG-Like CNN with Batchnorm code below and only add a couple things:
#
# 1. code to pass in run parameter from Domino
# 2. a timer to keep track of how long it takes to build the model 
# 3. a confusion matrix to see where improvements in accuracy are to be had
# 4. an export to JSON of accuracy and time so Domino runs can be easily compared from the UI
# 5. minor changes to make it work on TF 1.4.1 for the CUDA 8 / P2 test




# Fashion MNIST is a drop-in replacement for the very well known, machine learning hello world, MNIST dataset. It has same number of training and test examples and the images have the same 28x28 size and there are a total of 10 classes/labels, you can read more about the dataset here : [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
# 
# ## Approach
# 
# VGG Like Model With Batchnorm
# 
# Split the original training data into 80% training and 20% validation. This helps to see whether we're over-fitting on the training data.
# 
# The model is initially trained for 10 epochs and another 10 epochs with a lower learning late. After the initial 20 epochs, data augmentation is added, which generates new training samples by rotating, shifting and zooming on the training samples, and trained for another 50 epochs.
# 
# Also, to avoid hot encoding the labels, `sparse_categorical_crossentropy` is used when compiling the models.



# get run parameters

# data augementation epochs
import sys
da_epochs = int(sys.argv[1])



# Required Libaries

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
np.random.seed(12345)


# Download and Load Fashion-MNIST

train_images_path = keras.utils.get_file('train-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz')
train_labels_path = keras.utils.get_file('train-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz')
test_images_path = keras.utils.get_file('t10k-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz')
test_labels_path = keras.utils.get_file('t10k-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz')

# function to load the data
def load_mnist(images_path, labels_path):
    import os
    import gzip
    import numpy as np

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# load the data
X_train_orig, y_train_orig = load_mnist(train_images_path, train_labels_path)
X_test, y_test = load_mnist(test_images_path, test_labels_path)
X_train_orig = X_train_orig.astype('float32')
X_test = X_test.astype('float32')
X_train_orig /= 255
X_test /= 255


# create test and train for model validation while building
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=12345)



# Data Augmentation

batch_size = 512

# reshape
img_rows = 28
img_cols = 28
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

# define data augmentation details
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=batch_size)
val_batches = gen.flow(X_val, y_val, batch_size=batch_size)

# define the normalization details
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def norm_input(x): return (x-mean_px)/std_px



# VGG Like Model With Batchnorm

# define the model
vgg_cnn = keras.models.Sequential([
    keras.layers.Lambda(norm_input, input_shape=(28,28, 1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(10, activation='softmax')
])


# compile the model
vgg_cnn.compile(keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# build the model and time it
import time
t0 = time.time()

vgg_cnn.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(X_val, y_val))


vgg_cnn.optimizer.lr = 0.0001


vgg_cnn.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=1,
          validation_data=(X_val, y_val))


# Data Augmentation
vgg_cnn.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=da_epochs, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)


t1 = time.time()
timeit=t1-t0




# write out resutls
 
 
# make confusion matrix plot
def plot_confusion_matrix(cm, target_names, path, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
 
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
 
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
 
    title:        the text to display at the top of the matrix
 
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
 
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
 
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
 
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
 
    if cmap is None:
        cmap = plt.get_cmap('Blues')
 
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
 
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
 
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
 
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig(path + 'ConfMatx.png', format='png')
    plt.gcf().clear()
    
    
# score the model
test_loss, test_acc = vgg_cnn.evaluate(X_test, y_test)
predictions = vgg_cnn.predict(X_test)
preds_index = predictions.argmax(axis=1)


# labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# build conf matrx
import sklearn as sk
from sklearn import metrics
plot_confusion_matrix(cm           = metrics.confusion_matrix(y_test, preds_index), 
                      normalize    = False,
                      path         = "results/",
                      target_names = class_names,
                      title        = "Confusion Matrix")


# export json
import json
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Time": round(timeit, 3), "Acc": round(test_acc, 3)}))

