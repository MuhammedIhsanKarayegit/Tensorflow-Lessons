import cv2 as cv
import uuid
import os
import time

labels = ['thumbsup', 'thumbsdown']
number_img = 12

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

# Eğer dizin yoksa oluştur
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)

for label in labels:
    cap = cv.VideoCapture(0)
    print('Collectin images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_img):
        print('Collegting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, frame)
        cv.imshow('frame', frame)
        time.sleep(2)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)
    os.system(f"git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}")
    

cap.release()
cv.destroyAllWindows()










