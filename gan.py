# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:31:44 2021

@author: vedhs
"""
from tensorflow import keras
import tensorflow as tf

import numpy as np

import random

from PIL import Image
import glob
import  matplotlib.pyplot as plt


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


def define_generator():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(4, (4,4),strides=(2,2), padding='same',input_shape=(256,256,3)))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2D(8, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2D(16, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2D(32, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2DTranspose(32, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2DTranspose(16, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2DTranspose(8, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2DTranspose(4, (4,4),strides=(2,2), padding='same'))    
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.BatchNormalization(momentum=0.15, axis=-1))
    
    model.add(keras.layers.Conv2DTranspose(3, (4,4), padding='same',activation='sigmoid'))    
    
            
    

    return model


gen=define_generator()
#gen=keras.models.load_model('gen.h5')

dm=keras.models.load_model('dm6.h5')



def simLoss(y_true,y_pred):    
    
    return tf.reduce_mean(tf.abs(y_true-y_pred),axis=(1,2,3))+tf.reduce_mean(1-dm(y_pred))


es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1)
esg = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
itrn=101

#generated=np.load('gen3.npy')

for i in range(itrn): 
    print('iteration',i)

    
    x=get_random_real_imgs(300)
    gen.compile(optimizer='adam',loss=simLoss)

    
    gen.fit(x,x,epochs=100,shuffle=True,callbacks=[esg])
    xr=get_random_real_imgs(6)
    r=gen.predict(xr)
    
    
    fig=plt.figure()
    fig.add_subplot(2,3,1)
    plt.imshow(Image.fromarray(np.array(xr[0]*255,dtype=np.int8),'RGB'))
    fig.add_subplot(2,3,2)
    plt.imshow(Image.fromarray(np.array(xr[1]*255,dtype=np.int8),'RGB'))
    fig.add_subplot(2,3,3)
    plt.imshow(Image.fromarray(np.array(xr[2]*255,dtype=np.int8),'RGB'))
    fig.add_subplot(2,3,4)
    plt.imshow(Image.fromarray(np.array(r[0]*255,dtype=np.int8),'RGB'))
    fig.add_subplot(2,3,5)
    plt.imshow(Image.fromarray(np.array(r[1]*255,dtype=np.int8),'RGB'))
    fig.add_subplot(2,3,6)
    plt.imshow(Image.fromarray(np.array(r[2]*255,dtype=np.int8),'RGB'))
    plt.show()
    plt.pause(0.05)
    
    
    
    
    randn=5
    for j in range(randn):
        x=get_random_real_imgs(200)
        last=gen.predict(x)

        x=np.concatenate((monet_imgs,x,last))
        y=np.concatenate((np.ones(300),np.zeros(last.shape[0]+last.shape[0])))
        
        dm.fit(x,y,epochs=10,shuffle=True,callbacks=[es])
        
    
    gen.save('gen6/gen_'+str(i)+'.h5')
    del x,y,r
    
gen.save('gen.h5')



