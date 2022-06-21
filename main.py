import numpy as np
import os
import cv2
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

trainPath = 'Train'

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

NUM_CATEGORIES = len(os.listdir(trainPath))

imageData = []
imageLabels = []

for i in range(NUM_CATEGORIES):
    path = 'Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            imageFromArray = Image.fromarray(image, 'RGB')
            resizedImage = imageFromArray.resize((IMG_HEIGHT, IMG_WIDTH))
            imageData.append(np.array(resizedImage))
            imageLabels.append(i)
        except:
            print("Error image " + img)

imageData = np.array(imageData)
imageLabels = np.array(imageLabels)

shuffleIndexes = np.arange(imageData.shape[0])
np.random.shuffle(shuffleIndexes)
imageData = imageData[shuffleIndexes]
imageLabels = imageLabels[shuffleIndexes]

X_train, X_test, y_train, y_test = train_test_split(imageData, imageLabels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_test = X_test/255

y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_test = keras.utils.to_categorical(y_test, NUM_CATEGORIES)

print(y_train.shape)
print(y_test.shape)

model = keras.models.Sequential([    
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(43, activation='softmax')
])

learningRate = 0.001
epochs = 100

opt = Adam(lr=learningRate, decay=learningRate / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_test, y_test))

model.save("model.h5")
