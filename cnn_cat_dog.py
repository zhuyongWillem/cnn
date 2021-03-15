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
cnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#模型训练
cnn_model.fit_generator(training_set,epochs=20)

#训练集数据预测准确率
accuracy_train = cnn_model.evaluate_generator(training_set)
print(accuracy_train)

#模型存储
cnn_model.save('task1_model_1.h5')

#模型加载
from keras.models import load_model
model_new = load_model('task1_model_1.h5')

#测试数据集预测准确率
test_set = train_datagen.flow_from_directory('task1_data/training_set',target_size=(50,50),batch_size=32,class_mode='binary')
accuracy_test = cnn_model.evaluate_generator(test_set)
print(accuracy_test)

#单张图片的预测
from keras.preprocessing.image import load_img,img_to_array
pic_1 = '1.png'
pic_1 = load_img(pic_1,target_size=(50,50))
pic_1 = img_to_array(pic_1)
pic_1 = pic_1/255
pic_1 = pic_1.reshape(1,50,50,3)
result = cnn_model.predict_classes(pic_1)

print('dog' if result == 1 else 'cat')

fig2 = plt.figure()
plt.imshow(pic_1[0])
plt.show()

#本地九张图片处理
a = [i for i in range(1,10)]
fig3 = plt.figure(figsize=(10,10))
for i in a:
    img_name = str(i) + '.png'
    pic_1 = load_img(img_name, target_size=(50, 50))
    pic_1 = img_to_array(pic_1)
    pic_1 = pic_1 / 255
    pic_1 = pic_1.reshape(1, 50, 50, 3)
    result = cnn_model.predict_classes(pic_1)
    print('dog' if result == 1 else 'cat')
    plt.subplot(3,3,i)
    plt.imshow(pic_1[0])
    plt.title('predict result: dog' if result == 1 else 'predict result: cat')

plt.show()

