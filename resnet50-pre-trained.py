# Import libraries
import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds
import keras
from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras import Model

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# Get the full dataset (batch_size=-1) in NumPy arrays from the returned tf.Tensor object
cifar10_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN, batch_size=-1 ) 
cifar10_test = tfds.load(name="cifar10", split=tfds.Split.TEST, batch_size=-1)

# Convert tfds dataset to numpy array records
cifar10_train = tfds.as_numpy(cifar10_train) 
cifar10_test = tfds.as_numpy(cifar10_test)


# Seperate feature X and label Y
X_train, Y_train = cifar10_train["image"], cifar10_train["label"]
X_test, Y_test = cifar10_test["image"], cifar10_test["label"]
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=99, test_size=0.2)

# Normalize the image data
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

# Specify the input image size and number of classes
img_width, img_height = 32, 32
nb_classes = 10

# Converts a class vector (integers) to binary class matrix to have one-hot encoding label. For example:
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes) 
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)

# Initialize the model vgg16 from tf.keras.applications
model_resnet50 = tf.keras.applications.resnet50.ResNet50(
    weights='imagenet', # None for random initialization, or 'imagenet' for using pre-training on ImageNet. 
    include_top=False, # Whether to include the 3 fully-connected layers at the top of the network.
    input_shape=(224, 224, 3)) # Specify input input_shape

#Initialize the data augmentator
from keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(rotation_range=10, Width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_augmented = train_generator.flow(X_train, Y_train, batch_size=64, shuffle=True)

from keras.models import Sequential
from keras.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D
from keras.layers import UpSampling2D, Input

#I initialize a sequencial model and then add the resnet50 model as a layer. I choose this approach because it is more intuitive for me
model_cifar = Sequential()
model_cifar.add(Input(shape=(32, 32, 3)))

#If I input 32x32 images, the output feature map will have size 1x1 which is too small for the model to make predictions on. 
#I don't want to remove modules from the model since that may reduce its complexity so I decided to upsampling the cifar10 images to 224x224
model_cifar.add(UpSampling2D(input_shape=(32, 32, 3), size=(7,7)))
model_cifar.add(model_resnet50)

#I added this to reduce the number of params in the model while still retaining important information
#Without global average pooling the model has around 100M params which is too large to train in a timely manner and may cause overfitting
model_cifar.add(GlobalAveragePooling2D())

#This is the custom dense layers so that I can use resnet50 to predict cifar10 images
model_cifar.add(Flatten())
model_cifar.add(BatchNormalization())
model_cifar.add(Dense(2048, activation='relu'))
model_cifar.add(BatchNormalization())
model_cifar.add(Dropout(0.5))
model_cifar.add(Dense(1024, activation='relu'))
model_cifar.add(BatchNormalization())
model_cifar.add(Dropout(0.5))
model_cifar.add(Dense(10, activation='softmax'))

#I set the resnet50 to trainable
model_cifar.layers[1].trainable=True

model_cifar.summary()

# Compile the model
model_cifar.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

history = model_cifar.fit(train_augmented, epochs=20, validation_data = (X_val, Y_val), batch_size=64)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluate the model prediction on a data sample
pred = model_cifar.predict(X_test[:1]) # Make prediction on a data sample
print("Model prediction: " + str(pred)) # Model prediction
print("True label: " + str(Y_test[:1])) # True label

# Evaluate the model prediction on the entire test set
preds = model_cifar.predict(X_test) # Make prediction on the entire test set
preds_index = np.argmax(preds, axis=1) # Get the index of maximum class probability of each of the data sample
label_index = np.argmax(Y_test, axis=1) # Get the index of maximum class label

# Compare the predictions with the true labels
comparison_result = np.equal(preds_index, label_index) # Return the comparison result which is an array of True/False.

# Calculate the number of correct predictions (True values in the comparison result array).
correct_preds = comparison_result.sum() # Compute the sum of elements across dimensions of a tensor.

# Show accuracy
print("Number of correct predictions: " + str(correct_preds))
print("Test accuracy: " + str(correct_preds/X_test.shape[0]))

result = model_cifar.evaluate(X_test, Y_test)
print(model_cifar.metrics_names)
print("Loss and accuracy on the test set: loss = {}, accuracy = {}".format(result[0],result[1]))