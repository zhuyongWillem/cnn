# Description：
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/16 9:04

#加载第一张图片
from keras.preprocessing.image import load_img,img_to_array
pic_1 = '1.png'
pic_1 = load_img(pic_1,target_size=(224,224))

from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.imshow(pic_1)
plt.show()

pic_1 = img_to_array(pic_1)

#数据预处理
from keras.applications.vgg16 import preprocess_input
import numpy as np
x = np.expand_dims(pic_1,axis=0)
x = preprocess_input(x)

#核心特征提取
from keras.applications.vgg16 import VGG16
model_vgg = VGG16(weights='imagenet',include_top=False)
features = model_vgg.predict(x)

#flatten
features = features.reshape(1,7*7*512)

model_vgg = VGG16(weights='imagenet', include_top=False)


# define a method to load and preprocess the image
def modelProcess(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x_vgg = model.predict(x)
    x_vgg = x_vgg.reshape(1, 25088)
    return x_vgg


# list file names of the training datasets
import os

folder = "task2_data/cats"
dirs = os.listdir(folder)
# generate path for the images
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]

# preprocess multiple images
features1 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features1[i] = feature_i

folder = "task2_data/dogs"
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]
features2 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features2[i] = feature_i

# label the results
print(features1.shape, features2.shape)
y1 = np.zeros(300)
y2 = np.ones(300)

# generate the training data
X = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(X.shape, y.shape)

#数据分离
from sklearn.model_selection import train_test_split
x_trian,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#全连接层
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=10,activation='relu',input_dim=25088))
model.add(Dense(units=1,activation='sigmoid'))

#核心参数配置
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_trian,y_train,epochs=50)

#表现评估
from sklearn.metrics import accuracy_score
y_train_predict = model.predict_classes(x_trian)
accuracy_train = accuracy_score(y_train,y_train_predict)
y_test_predict = model.predict_classes(x_test)
accuracy_test = accuracy_score(y_test,y_test_predict)
print(accuracy_train,accuracy_test)

#图片加载》图片格式转化》数据预处理》vgg16特征提取》模型预测

