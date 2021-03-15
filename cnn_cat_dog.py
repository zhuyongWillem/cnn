# Description：
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/14 16:54

#数据加载
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('task1_data/training_set',target_size=(50,50),batch_size=32,class_mode='binary')

#查看数据类型
print(type(training_set))

#加载图片名称
print(training_set.filenames)

#确认标签
print(training_set.class_indices)

#train_set[][][]批次，x or y, 第几个样本
print(training_set[0][1])

#第一个批次第一个样本的输入数据
print(training_set[0][0][0,:,:,:].shape)

#可视化
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.imshow(training_set[0][0][0,:,:,:])
plt.show()

#加载后按批次存放的每个样本对应的索引号
print(training_set.index_array)

#获取文件名称
print(training_set.filenames[2384])

#建立CNN模型
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
cnn_model = Sequential()
cnn_model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2)))
cnn_model.add(Conv2D(32,(3,3),activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=128,activation='relu'))
cnn_model.add(Dense(units=1,activation='sigmoid'))
cnn_model.summary()

#模型配置
cnn_model.compile()