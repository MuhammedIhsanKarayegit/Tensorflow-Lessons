import cv2 as cv
import os
import time
import uuid

labels = ['Thumbs Up']
number_imgs = 5

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.makedirs(IMAGES_PATH, exist_ok=True)
    elif os.name == 'nt':
        os.makedirs(IMAGES_PATH, exist_ok=True)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

"""
for label in labels:
    cap = cv.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting Images {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, frame)
        cv.imshow('frame', frame)
        time.sleep(2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
"""

label_img_path = r"C:\\Users\\MiKar\\OneDrive\\Documents\\GitHub\\Tensorflow-Lessons\\Tensorflow\\labelImg"

if os.name == 'nt':
    # Windows için
    os.system(f'cd "{label_img_path}" && pyrcc5 -o libs/resources.py resources.qrc')

os.system(f"cd {label_img_path} && python labelImg.py")


