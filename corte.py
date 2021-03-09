import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

import PIL
from PIL import Image
from sklearn.model_selection import train_test_split

def ler_png(path):
    img = Image.open(path).convert('P')
    WIDTH, HEIGHT = img.size
    if WIDTH != 240:
        img = img.crop((8, 8, 248, 248))
    if WIDTH > 256:
        img.thumbnail((240,240), PIL.Image.ANTIALIAS)
    return np.asarray(img)

def load_image_dataset(path_dir):

        images = []
        os.chdir(path_dir)
        for file in glob.glob("*.png"):
                img = ler_png(file)
                if np.any(img != 0):
                    images.append(img)

        return (np.asarray(images))
        
        
dir1 = 
dir2 = 
dir3 = 
train_images1 = load_image_dataset(dir1)
train_images2 = load_image_dataset(dir2)
train_images3 = load_image_dataset(dir3)



train_labels1 = np.repeat(0,train_images1.shape[0])
train_labels2 = np.repeat(1,train_images2.shape[0])
train_labels3 = np.repeat(1,train_images3.shape[0])



treinoMTA = np.concatenate((train_images1, train_images2))
labelMTA = np.concatenate((train_labels1, train_labels2))

treinoErica = np.concatenate((train_images1, train_images3))
labelErica  = np.concatenate((train_labels1, train_labels3))

treinoMTA = treinoMTA/255.0
treinoErica = treinoErica/255.0


treinoMTA = tf.convert_to_tensor(treinoMTA)
treinoErica = tf.convert_to_tensor(treinoErica)

treinoMTA = tf.reshape(treinoMTA, [559,240,240,1])
treinoErica = tf.reshape(treinoErica, [553,240,240,1])


model = keras.Sequential([
    keras.layers.Conv2D(32,(5,5),activation = 'relu',padding='same', input_shape = (240,240,1)),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32,(3,3),activation = 'relu',padding='same'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(treinoMTA, labelMTA, epochs=20)

model1 = keras.Sequential([
    keras.layers.Conv2D(32,(5,5),activation = 'relu',padding='same', input_shape = (240,240,1)),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32,(3,3),activation = 'relu',padding='same'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)
model1.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
              
model1.fit(treinoErica, labelErica, epochs=20)


probs  = []
probs1 = []

#Parte dos testes em novas imagens
dir4 = 
for i in range(12):
    pasta = dir4 + str(i+1)
    Teste2 = load_image_dataset(pasta)
    Teste2 = Teste2/255.0
    Teste2 = tf.convert_to_tensor(Teste2)
    Teste2 = tf.reshape(Teste2, [Teste2.shape[0],240,240,1])
    
    previsao = model.predict(Teste2)
    previsao1 = model1.predict(Teste2)

    for j in  range(previsao.shape[0]):
        probs.append(previsao[j,1])

    for j in  range(previsao1.shape[0]):
        probs1.append(previsao1[j,1])
        
    probs  = np.asarray(probs)
    probs1 = np.asarray(probs1)

    index  = np.argmax(probs)
    index1 = np.argmax(probs1)
    
    
    Teste2 = tf.reshape(Teste2, [Teste2.shape[0],240,240])
    Teste2.numpy()

    #dir5 = 
    #os.chdir(dir5)
    plt.imshow(Teste2[index], cmap = 'gray')
    #plt.savefig(str(i+1) + '.png', format='png')
    plt.show()
    
    #dir6 = 
    #os.chdir(dir6)
    plt.imshow(Teste2[index1], cmap = 'gray')
    #plt.savefig(str(i+1) + '.png', format='png')
    plt.show()

    probs  = []
    probs1 = []
    
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


model_json = model1.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("model1.h5")
