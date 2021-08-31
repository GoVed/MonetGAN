# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:11:20 2021

@author: vedhs
"""
from tensorflow import keras

import numpy as np

import random

from PIL import Image
import glob

#getting the data
monet_imgs=[]
images=glob.glob("monet_jpg/*.jpg")
for image in images:
    monet_imgs.append(np.array(Image.open(image)))
monet_imgs=np.array(monet_imgs,dtype=np.float16)
monet_imgs/=255.0

def get_random_real_imgs(n=300):
    real_imgs=[]
    all_images=glob.glob("photo_jpg/*.jpg")
    images=[]
    for i in range(n):
        randid=random.randint(0,len(all_images)-1)
        images.append(all_images.pop(randid))
    
    for image in images:
        real_imgs.append(np.array(Image.open(image)))
    real_imgs=np.array(real_imgs,dtype=np.float16)
    real_imgs/=255.0
    return real_imgs


#making the dis model
dm=keras.models.Sequential()

dm.add(keras.layers.Conv2D(8, (4,4), strides=(2,2), padding='same',input_shape=(256,256,3)))
dm.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
dm.add(keras.layers.LeakyReLU(alpha=0.2))

dm.add(keras.layers.Conv2D(16, (4,4), strides=(2,2), padding='same'))
dm.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
dm.add(keras.layers.LeakyReLU(alpha=0.2))

dm.add(keras.layers.Conv2D(32, (4,4), strides=(2,2), padding='same'))
dm.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
dm.add(keras.layers.LeakyReLU(alpha=0.2))

dm.add(keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same'))
dm.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
dm.add(keras.layers.LeakyReLU(alpha=0.2))

dm.add(keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
dm.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
dm.add(keras.layers.LeakyReLU(alpha=0.2))


dm.add(keras.layers.Flatten())
dm.add(keras.layers.Dense(1,activation='sigmoid'))


dm.compile(optimizer='adam',loss='binary_crossentropy')

generated=np.load('gen4.npy')
batches = 3
for i in range(batches):
    x=np.concatenate((monet_imgs,get_random_real_imgs(),generated))
    y=np.concatenate((np.full((300),1),np.full((300+generated.shape[0]),0)))
    dm.fit(x,y,epochs=10,shuffle=True)
    
dm.save('dm6.h5')
