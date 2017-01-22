import keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import load_img,img_to_array
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping,History
from keras.preprocessing.image import load_img
from keras.models import model_from_json
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from numpy.random import randint



def roi(img):
    #img =img[55:140,30:290]
    img =img[40:img.shape[0]-25,:]
    #img = img[60:140,40:280]
    return cv2.resize(img,(200,66), interpolation=cv2.INTER_AREA)

def normalization(img):
    img = img/127.5-1.0
    img = img.astype(np.float32)
    return img

def augment_brightness(img):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2HSV)
    random_bright = 0.25+np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def transpose_image(img,steering):
    img = cv2.flip(img,1)
    return img,-1.0*steering


def image_generator(driving_log):
    driving_log = driving_log.sample(frac=1).reset_index(drop=True)

    for index, row in driving_log.iterrows():



        #Select Left,Center,Right image
        sel_lcr = np.random.randint(3)

        if sel_lcr==0:
            fname = os.path.basename(row['left'])
            steering = np.float32(row['steering']) + 0.25
            img = load_img('IMG/'+fname)
            img =np.array(img)
        elif sel_lcr==1:
            fname = os.path.basename(row['center'])
            steering = np.float32(row['steering'])
            img = load_img('IMG/'+fname)
            img =np.array(img)
        else:
            fname = os.path.basename(row['right'])
            steering = np.float32(row['steering'])  - 0.25
            img = load_img('IMG/'+fname)
            img =np.array(img)


        #Crop and Resize the image
        img = roi(img)

        #Normalize the image
        img = normalization(img)

        #Add Random Brightness
        aug_bright = np.random.randint(3)

        if aug_bright ==0:
            img = augment_brightness(img)
        else:
            pass

        #Change the color space
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2YUV)

        #Random Flip
        trans = np.random.randint(2)

        if trans ==0:
            img,steering = transpose_image(img,steering)

        else:
            pass

        #Reshape the image
        img = np.reshape(img,(3,66,200))

        yield img, steering



def batch_generator(driving_log,  batch_size=32, *args, **kwargs):
    num_rows = len(driving_log.index)
    train_images = np.zeros((batch_size, 3, 66, 200))
    train_steering = np.zeros(batch_size)
    ctr = None
    while 1:
        for j in range(batch_size):
            # Reset generator if over bounds
            if ctr is None or ctr >= num_rows:
                ctr = 0
                images = image_generator(driving_log,  *args, **kwargs)
            train_images[j], train_steering[j] = next(images)
            ctr += 1
        yield train_images, train_steering


driving_log = pd.read_csv("driving_log.csv").reset_index()
print("Number of Original Data",len(driving_log))

#Cut off 75% of low steering angle
num_drops = int(len(driving_log[np.abs(driving_log["steering"])<=0.1])*0.75)
drop_lows = driving_log[driving_log["steering"]==0]["index"].values[0:num_drops]

#Shuffle the data
driving_log_new = driving_log.drop(drop_lows,axis=0).sample(frac=1.0)
print("New Data",len(driving_log_new))

#Divide the data into training(80%) and validation(20%) data
num_training=(int(len(driving_log_new)*0.8))

training_data = driving_log_new[0:num_training]

print("Num of Training data",len(training_data))
validation_data = driving_log_new[num_training:]
print("Num of Validation data",len(validation_data))

#Make dataset
train_data = batch_generator(training_data)
val_data = batch_generator(validation_data)

#Nvidia Model
model = Sequential()
#model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(3,66,200),name="Normalization"))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', name='Conv1',input_shape=(3,66,200)))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', name='Conv2'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', name='Conv3'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv4'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv5'))
model.add(Flatten())
#model.add(Dense(1164, activation='relu', name='FC1'))
#model.add(Dropout(0.4))
model.add(Dense(100, activation='relu', name='FC1'))
#model.add(Dropout(0.5))
model.add(Dense(50, activation='relu', name='FC2'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu', name='FC3'))
#model.add(Dropout(0.5))
model.add(Dense(1, name='output'))
model.summary()

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])


model_json = model.to_json()
model_name = 'model'
h = model.fit_generator(train_data, validation_data = val_data,
                            samples_per_epoch = 28000,#28000
                            nb_val_samples = 2800,
                            nb_epoch=2, verbose=1)


with open(model_name+'.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_name+'.h5')

with open('./model.json', 'r') as json_file:
    model = model_from_json(json_file.read())
#save_model(model_json,model_name)
model.load_weights('./model.h5')
model.summary()