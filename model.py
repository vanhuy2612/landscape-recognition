import os
import sys
import numpy as np
import tensorflow
import keras
import cv2, random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils, to_categorical
from keras.datasets import mnist
print("tensorflow version: ", tensorflow.__version__)
print("keras version: ", keras.__version__)

objective = 'categorical_crossentropy'
adam = keras.optimizers.Adam(lr=0.0001) # thuật toán training thông qua tham số  

# format image
HEIGHT = 256
WIDTH = 256
DEEP = 3 # nó biểu thị cho màu RGB

# Training model có nghĩa là tìm ra tập trọng số (network weights) tốt nhất của model để dự đoán.
# khởi tạo network weights bằng các số ngẫu nhiên nhỏ (từ 0 đến 0.5) bằng cách sử dụng uniform distribution (‘uniform‘). 
# Một cách khác thường dùng để khởi tạo network weights  là Gaussian distribution.
def Model():
    model = Sequential() # Khởi tạo 1 model trong keras, add - method 
    model.add(Conv2D(32,kernel_size=(3,3), strides = 1,padding ='same',input_shape=(HEIGHT,WIDTH,DEEP),activation='relu'))
    model.add(Conv2D(32,kernel_size=(3,3), strides = 1,padding ='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Conv2D(64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), strides = 1,padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), strides = 1, padding='same', activation='relu'))
#     model.add(Convolution2D(128, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten dùng để lát phằng layer (convert thành mảng 1 chiều), vd : shape : 20x20 qua layer này sẽ là 400x1
    model.add(Flatten())

    # Dense là 1 lớp neurol net work bình thường (đầu vào phải có rank = 1, tức mảng 1 chiều)
    # 256 - số chiều không gian đầu ra = số neurol , 
    # nếu ko có activation thì kích hoạt tuyến tính : a(x) = x
    model.add(Dense(256, activation='relu'))
    
    # Làm giảm overfitting (chống học quá vừa vặn)
    # kỹ thuật dropout là việc chúng ta sẽ bỏ qua một vài unit trong suốt quá trình train trong mô hình, 
    # những unit bị bỏ qua được lựa chọn ngẫu nhiên. 
    # Ở đây, chúng ta hiểu “bỏ qua - ignoring” là unit đó sẽ không tham gia 
    # và đóng góp vào quá trình huấn luyện (lan truyền tiến và lan truyền ngược).
    #Về mặt kỹ thuật, tại mỗi giai đoạn huấn luyện, 
    # mỗi node có xác suất bị bỏ qua là 1-p và xác suất được chọn là p

    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # vì mỗi ảnh sẽ thuộc class từ 0->2 nên output layer sẽ có 3 node để tính phần trăm ảnh là 0,1,2
    #
    model.add(Dense(3))

    # activation softmax dùng trong đa phân loại, dùng softmax function dùng để chuyển đổi giá trị thực 
    # trong các node ở output layer sang giá trị phần trăm
    # nếu chỉ có 2 class đầu ra thì dùng sigmoid, >2 dùng softmax.
    # sigmoid trên output layer để đảm bảo rằng output chỉ nằm trong khoảng 0 và 1 với threshold là  0.5 để dễ dàng mapping kết quả của network là 1

    model.add(Activation('softmax'))

    # Biên tập lại toàn bộ model :
    # optimizer: thuật toán training, function loss, metrics : hiển thị khi model được training
    # loss function : chuyển dạng one-hot encoding giá trị dự đoán ở output layer sau hàm softmax function cùng kích thước
    #       sang vector cùng kích thước Ex: [0.12 0.5 0.13] => [0 1 0]
    # thuật toán tối ưu Adam trong file README
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

