# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:41:03 2019

@author: Abdulrahman Alothman
"""

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.layers import Dense ,Flatten
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
from tensorflow.python.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.layers import  GlobalAveragePooling2D, BatchNormalization
import os

script_dir = os.path.dirname(".")
print(os.listdir("train_set"))

train_set_path=os.path.join(script_dir,'train_set')
validation_set_path = os.path.join(script_dir, 'train_set')
test_set_path = os.path.join(script_dir, 'test_set')


resnet_weights_path = 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #Resnet50 pretrained weights

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False  # 0. katmanın ağırlıklarını güncellenmeyecek

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
count = sum([len(files) for r, d, files in os.walk("train_set/")])        #train resimlerin sayısı
for l in model.layers:
    print(l.name, l.trainable)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.10)

test_datagen = ImageDataGenerator(rescale=1./ 255)

image_size = 224
batch_size = 10
input_size = (224, 224)


training_set = train_datagen.flow_from_directory(train_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='categorical')

validation_set = validation_datagen.flow_from_directory(validation_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='categorical')


test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            color_mode="rgb",
                                            shuffle = False,
                                            batch_size=1,
                                            class_mode='categorical')


print(training_set.class_indices)
print('You Have :',len(training_set.class_indices),'Class')


model.summary()

model_info = model.fit_generator(training_set,
                         steps_per_epoch=count//batch_size,
                         epochs=10,
                         validation_data=validation_set,
                         validation_steps=378//batch_size)

model.save_weights('my_model_weights.h5')  #save model

#model.load_weights('my_model_weights.h5') #load model


scoreSeg = model.evaluate_generator(test_set,400)
test_set.reset()
predict = model.predict_generator(test_set,400)


print('***tahmin Değerleri***')
print(np.argmax(predict, axis=1))
print('***Gerçek Değerleri***')
print(test_set.classes)

print(test_set.class_indices)

pred=np.argmax(predict, axis=1)

print("Confusion Matrix")
print(confusion_matrix(test_set.classes,pred))


print("Results")
print(classification_report(test_set.classes,pred,target_names=(sorted(test_set.class_indices.keys()))))

from sklearn.metrics import accuracy_score
print ('Accuracy:', accuracy_score(test_set.classes, np.argmax(predict, axis=1)))

print('The testing accuracy is :',scoreSeg[1]*100, '%')
