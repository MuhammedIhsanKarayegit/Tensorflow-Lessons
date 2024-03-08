import cv2 as cv
import os
import time
import uuid

labels = ['Thumbs Up']
number_imgs = 5

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

#Vermiş olduğumuz path değişkenine göre gerekli dosyayı işletim sistemine uygun bir şekilde kaydediyoruz.
if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.makedirs(IMAGES_PATH, exist_ok=True)
    elif os.name == 'nt':
        os.makedirs(IMAGES_PATH, exist_ok=True)
#Oluşturmuş oludğumuz label listesi içinde dönüp liste içindeki isimlerle dosyaları oluşturuyoruz.
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

#Bu kısımda kamerdan number_img sayısı kadar fotoğraf çekimi gerçekleştiriyoruz her iki saniyede bir.
#Ardından bu fotoğrafları uuid kütüphanesi sayesinde rastegele bir şekilde isimlendirekek daha önceden oluşturmuş olduğumuz
#dosyalara içine isimlerine göre ayırarak kaydediyoruz. 
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

#labelImg uygulamasını kullanmak için gerekli yol sağlandı ardından bu yoldan yola çıkarak uygulamanın çalışmasını gerçekliyoruz.
#Bu uygulamanın amacı çekmiş olduğumuz fotoğraflardaki tanımlamak istedğimiz nesneleri belirtelek bu nesnelere gerekli tag ları ekliyoruz.
#Tag işlemi tamamlandıktan sonra makinemizi eğitme aşamasına geçiyoruz. Makineyi eğitmek için gerekli olan kodlar Traing_and_Detection.py 
#isimli dosya içinde bulunmaktadır.

label_img_path = r"C:\\Users\\MiKar\\OneDrive\\Documents\\GitHub\\Tensorflow-Lessons\\Tensorflow\\labelImg"

if os.name == 'nt':
    # Windows için
    os.system(f'cd "{label_img_path}" && pyrcc5 -o libs/resources.py resources.qrc')

os.system(f"cd {label_img_path} && python labelImg.py")


