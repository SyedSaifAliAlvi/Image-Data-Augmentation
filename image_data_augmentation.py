from skimage import transform
from skimage.util import random_noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import random

def data_augment(X_data,y):
  x = X_data.copy()
  y_new = y.copy()
  for i in X_data:
    k1 = np.fliplr(i)
    x.insert(0,k1)
    y_new.insert(0,1)

    k2 = np.flipud(i)
    x.insert(0,k2)
    y_new.insert(0,1)

    k3 = transform.rotate(i,random.uniform(-20,20))
    x.insert(0,k3)
    y_new.insert(0,1)

    k4 = random_noise(i,mode='salt',clip='True') 
    x.insert(0,k4)
    y_new.insert(0,1)
    
    k5 = random_noise(i,mode='gaussian',clip='True') 
    x.insert(0,k5)
    y_new.insert(0,1)

    k6 = random_noise(np.flipud(i),mode='salt',clip='True') 
    x.insert(0,k6)
    y_new.insert(0,1)

    k7 = random_noise(np.fliplr(i),mode='salt',clip='True') 
    x.insert(0,k7)
    y_new.insert(0,1)

    k8 = random_noise(random_noise(np.fliplr(i),mode='gaussian',clip='True'),mode='salt',clip='True') 
    x.insert(0,k8)
    y_new.insert(0,1)

    k9 = transform.rotate(i,random.uniform(-90,110))
    x.insert(0,k9)
    y_new.insert(0,1)

    k10 = random_noise(transform.rotate(i,random.uniform(-90,110)),mode='gaussian',clip='True') 
    x.insert(0,k10)
    y_new.insert(0,1)


  return x,y_new

def resize(img):
  image = Image.open(img)
  image = image.resize((128,128),Image.ANTIALIAS)
  return image

def imageToNumpyArray(img):
  N_array = np.asarray(img)
  return N_array

def toThreeChannel(image):
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img2 = np.zeros_like(img)
  img2[:,:,0] = gray
  img2[:,:,1] = gray
  img2[:,:,2] = gray
  cv2.imwrite(image, img2)

def convertImagesToArray(path):
  img_array = []
  for image in glob.glob(path):
    toThreeChannel(image)
    R_img = imageToNumpyArray(resize(image))
    img_array.append(R_img)
  return img_array

imageFolderPath = "/content/*.jpg"
image = convertImagesToArray(imageFolderPath)

y=[]
X,y = data_augment(image,y)

#Number of images in folder * 11 = Number output images
for i in range(11):
  plt.imshow(X[i])
  plt.show()

