import os
import numpy as np
import tensorflow
import keras
import cv2, random
import h5py
import matplotlib.pyplot as plt
from keras.models import Sequential
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

# chuyển label đầu ra thành ma trận Y dạng: [1,0,0] city, [0,1,0] forest, [0,0,1] sea
# train_labels_1hot - Ouput Y
one_hot_labels = np.zeros((len(train_label),3))
for i in range(len(train_label)) :
  one_hot_labels[i][train_label[i]] = 1
train_labels_1hot = one_hot_labels
train_labels_1hot.shape

# load data  
print("Loading data .................")
train_image = load_data(train_image_path)
test_image = load_data(test_image_path)

label_name = ['city', 'forest', 'sea']

# Create instance of Model
VGG13 = Model()
# summary: method giúp chúng ta tổng hợp model có bao nhiêu layer, tổng số tham số và shape của mỗi layer.
VGG13.summary()

"""#Tranning
+ Train_image
+ Train_label
"""

print(train_image.shape, train_label.shape, train_labels_1hot.shape)
# Start trainning:
# fit() ???????????????????????????????-------------------------------------------------------------
print("Start tranining:...............")
# thông thường để epochs càng lâu càng tốt. before epochs = 6. 
# thực hiện nhiều lần việc train test để chọn ra được param tốt nhất.
# train_image: đầu vào x , train_labels_1hot: output Y
# hàm dùng để cập nhật trọng số.

VGG13.fit(train_image,train_labels_1hot, batch_size=16,  epochs= 6, validation_split=0.1, shuffle=True)

# # Lưu kết quả training vào file trained_model.h5: ma trận trọng số.

# print("Create and Write result to file trained_model.hdf5")
# VGG13.save('trained_model_3v.hdf5')
# # open('trained_model.h5', 'w') as FOUT:
# # np.savetxt(FOUT, a_test)

# # Lấy trọng số . Ex y = ax + b thì nó sẽ tính ra a, b
# print("Get weights from model........................")
# VGG13.get_weights()