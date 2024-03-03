import cv2 as cv
import uuid
import os
import time

labels = ['thumsup']
number_img = 20

NEW_DIRECTORY_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

# Eğer dizin yoksa oluştur
if not os.path.exists(NEW_DIRECTORY_PATH):
    os.makedirs(NEW_DIRECTORY_PATH)

for label in labels:
    path = os.path.join(NEW_DIRECTORY_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)



