import os
import numpy as np
import tensorflow
import keras
import cv2, random
import matplotlib.pyplot as plt
import model as Model
from keras.models import Sequential, load_model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils, to_categorical
from keras.datasets import mnist

from model import Model
print("tensorflow version: ", tensorflow.__version__)
print("keras version: ", keras.__version__)

# os.listdir('drive/My Drive/Scene Classification')

# !unzip -q 'drive/My Drive/Scene Classification/data.zip' -d scene

# link data
city_path = './data/city'
forest_path = './data/forest'
sea_path = './data/sea'

# format image
HEIGHT = 256
WIDTH = 256
DEEP = 3 # nó biểu thị cho màu RGB

# link image
city_image_path = [os.path.join(city_path,i) for i in os.listdir(city_path)]
forest_image_path = [os.path.join(forest_path,i) for i in os.listdir(forest_path)]
sea_image_path = [os.path.join(sea_path,i) for i in os.listdir(sea_path)]

# init label city:0, forest:1, sea:2
city_label = [0 for i in range(len(city_image_path))]
forest_label = [1 for i in range(len(forest_image_path))]
sea_label = [2 for i in range(len(sea_image_path))]

print(len(city_image_path), len(forest_image_path), len(sea_image_path))

#path and label for training 

train_image_path = city_image_path[:480] + forest_image_path[:480] + sea_image_path[:480]   # total image 0 -> 480
train_label = city_label[:480] + forest_label[:480] + sea_label[:480]                       # total label

#path and label for test

test_image_path = city_image_path[480:] + forest_image_path[480:] + sea_image_path[480:]    # data test 480 -> hết

print(len(train_image_path), len(train_label), len(test_image_path))

# Shuffle data (Tron du lieu)
# list() convert and return a list
# zip(iterator1, iterator2, ...) ghép các phần tử Ex: 
# a = ("John", "Charles", "Mike")
# b = ("Jenny", "Christy", "Monica") 
# x = zip(a, b) => x: (('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))

z = list(zip(train_image_path, train_label)) 
# Trộn dữ liệu trong z
random.shuffle(z)
# unzip lấy lại dữ liệu train_image_path, train_label
train_image_path, train_label  = zip(*z)

# function đọc, convert ảnh theo 1 quy chuẩn
def read_image(image_path):
  # đọc ảnh ( giá trị trả về chính là ảnh đó), nếu muốn giá trị trả về là ảnh xám thì cv2.imread(path, 0)
  image = cv2.imread(image_path)
  # Chuyển đổi hình ảnh sang 1 không gian màu khác.
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  # interpolation: phép nội suy: NEAREST, LINEAR, AREA, CUBIC: nội suy hai chiều trên vùng lân cận 4 × 4 pixel, LANCZOS4
  return cv2.resize(image,(HEIGHT,WIDTH),interpolation = cv2.INTER_CUBIC)

# function trả về 1 danh sách ảnh đã theo 1 quy chuẩn từ đường dẫn vào.
def load_data(list_path):
  image_list= []
  for i, image_path in enumerate(list_path):
    image = read_image(image_path)
    image_list.append(image)
    if i % 500 == 0 : print('load {} in {} images'.format(i,len(list_path)))
  return image_list

# one_hot_labels = np.zeros((len(train_label),3))
# for i in range(len(train_label)) :
#   one_hot_labels[i][train_label[i]] = 1
# train_labels_1hot = one_hot_labels
# train_labels_1hot.shape

# load data  
print("Loading data .................")
# train_image = load_data(train_image_path)
test_image = load_data(test_image_path)

label_name = ['city', 'forest', 'sea']


print("Starting test image .......................................")

# Lấy dữ liệu đã train lên để test
VGG13 = Model()
VGG13.load_weights('trained_model_1v.hdf5')

def test(index):
  # predict() sử dụng mô hình để dự đoán ảnh đầu vào.
  # predict() hoạt động ntn ???????????????????======================================================
  predict = VGG13.predict(test_image[index].reshape((1,HEIGHT,WIDTH,DEEP)))
  plt.imshow(test_image[index])
  print("Du doan nhan cua anh:")
  print(predict)
  
  if np.argmax(predict) == 0: 
    print('I am sure this is city')
    plt.show()
  elif np.argmax(predict) == 1:
    print('I am sure this is forest')
    plt.show()
  else:      
    print('I am sure this is sea')
    plt.show()
  # Kết quả dự đoán là 1 ma trận xác xuất của 3 class city, forest, sea
  print(predict)
test(0)

# from keras.models import load_model
# # Pass your .h5 file here
# model = load_model('trained_model.hdf5')
# test_data = np.random.rand(1,300,26)
# result = model.predict(test_data)
# print (result)

