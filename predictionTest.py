import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

IMG_HEIGHT = 30
IMG_WIDTH = 30

modelPath = "model.h5"
loadedModel = tf.keras.models.load_model(modelPath)

image = cv2.imread("image.png")

imageFromArray = Image.fromarray(image, 'RGB')
resizeImage = imageFromArray.resize((IMG_HEIGHT, IMG_WIDTH))
expandInput = np.expand_dims(resizeImage,axis=0)
inputData = np.array(expandInput)
inputData = inputData/255

pred = loadedModel.predict(inputData)
result = pred.argmax()
print(result)