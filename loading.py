from google.colab import drive
drive.mount('/content/gdrive')

# !unzip '/content/gdrive/MyDrive/train.zip' -d 'train_data'
# !unzip '/content/gdrive/MyDrive/MDST/train.zip' -d 'train_data'

import os
import cv2

Xlist = []
ylist = []

folder_dir = "./train_data/train"

# Formatting all images into X
for images in os.listdir(folder_dir):
    
    # Formatting cat or dog : 0 or 1 into list y
    if images[0:3] == "cat":
        ylist.append(0)
    else:
        ylist.append(1)
    
    image = cv2.imread("./train_data/train/"+images)
    #print("/train_data/train/"+images)
    resized = cv2.resize(image, (80,80))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    final = gray/255
    Xlist.append(final)

import numpy as np

X = np.array(Xlist).reshape(len(Xlist), 80,80,1)
y = np.array(ylist)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Cast X and y to a numpy array. 
# X must be of a certain shape so use X = np.array(X).reshape(num_train_img, 80,80,1)
# The two middle dimensions 80 x 80 is the size of our image and we can change that if needed

X_train = np.array(X_train).reshape(len(X_train), 80,80,1)
y_train = np.array(y_train)

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Looking at data
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow( Xlist[i], cmap ="gray")
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
plt.show()

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
# add dense layers on top
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(1, activation = 'sigmoid'))

#Compile and train the model2
model2.compile(optimizer='adam',
            loss= "binary_crossentropy",
            metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs=9, validation_split=0.2,batch_size=40)

#Evaluate the model2
plt.plot(history2.history['accuracy'], label='accuracy')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model2.evaluate(X_test,  y_test, verbose=2)

model.summary()