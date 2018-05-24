from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.utils import shuffle
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

batch_size = 32
num_classes = 7
epochs = 15

# input image dimensions
# img_rows, img_cols = 200, 200

# num_classes = 10
img_rows, img_cols = 64, 64
# path2 = './imgfolder_b'
path2 = './myges_bin'
output = ["CALL", "COMB", "HI","NOTHING", "PUNCH", "STOP", "UP"]

def modlistdir(path):
  listing = os.listdir(path)
  retlist = []
  for name in listing:
    #This check is to ignore any hidden files/folders
    if name.startswith('.'):
        continue
    retlist.append(name)
  return retlist

def initialize():
  # the data, split between train and test sets
  imlist = modlistdir(path2)
  imlist = sorted(imlist)
  image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
  image1 = cv2.resize(image1, (img_rows, img_cols)) 

  m,n = image1.shape[0:2] # get the size of the images
  total_images = len(imlist) # get the 'total' number of images
  
  # create matrix to store all flattened images
  immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                       for images in imlist], dtype = 'f')
  immatrix = []
  for images in imlist:
    tmp_img = Image.open(path2 + '/' + images).convert('L')
    tmp_img = np.array(cv2.resize(np.array(tmp_img), (img_rows, img_cols)))
    tmp_img = tmp_img.flatten()
    immatrix.append(tmp_img)

  immatrix = np.array(immatrix, dtype='f')
  print (immatrix.shape)
  

  label=np.ones((total_images,),dtype = int)

  samples_per_class = total_images // num_classes
  print ("samples_per_class - ",samples_per_class)
  s = 0
  r = samples_per_class
  for classIndex in range(num_classes):
      print(classIndex, imlist[s], imlist[r-1])
      label[s:r] = classIndex
      s = r
      r = s + samples_per_class
  
  
  data,Label = shuffle(immatrix,label, random_state=2)
  train_data = [data,Label]
   
  (X, y) = (train_data[0],train_data[1])
   
   
  # Split X and y into training and testing sets
   
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return x_train, y_train, x_test, y_test

class GestureModel():
  def __init__(self, input_shape, weights = None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    if weights:
      model.load_weights(weights)
    self.model = model

  def train(self, x_train, y_train, x_test, y_test):
    self.model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
  def save(self):
    self.model.save_weights("bin_model_1.h5")

  def predict(self, img):
    img = cv2.resize(np.array(img), (img_rows, img_cols))
    img = img.reshape(1, img_rows, img_cols, 1)
    return output[np.argmax(self.model.predict(img)[0])]

  def evaluate(self, x_test, y_test):
    score = self.model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
  x_train, y_train, x_test, y_test = initialize()
  gestureModel = GestureModel(input_shape = (img_rows, img_cols, 1))#, weights = "bin_model.h5")
  gestureModel.train(x_train, y_train, x_test, y_test)
  gestureModel.evaluate(x_test, y_test)
  gestureModel.save()
  # img = Image.open("test/up1000.png")
  # print(gestureModel.predict(img))