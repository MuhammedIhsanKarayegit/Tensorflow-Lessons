import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

# İlk olarak Fashion MNIST datasetini tensorflow içinden yükleyerek başlıyoruz.
# Çekmiş olduğumuz bu dataset'i numpy içinden dört farklı array içine atıyoruz.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Gelen datasat içindeki resimleri adlandırmak için gerekli tag'ları (liste şeklinde) oluşturuyoz.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Veri seti içindeki ilk elmanı ekrana yazdırıyoz.
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""


# Elimizde bulunan eğitim array'lerini işlemek için (0 ve 1 arasında olması için) gerekli işlemleri gerçekleştiriyoruz.
train_images = train_images / 255.0
test_images = test_images / 255.0


 # Verilerin gerekli şekilde eğitilip eğitilmediğini kontrol etmek amacıyla ilk 25 veriyi ekrana çıkartıyoruz.
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

# Sinir ağlarını oluşturmak için basitleştirilmiş katmanlara ihtiyaç duyuyoruz aşağıda elimizdeki verileri daha basit katmanlara dönüştürüyoruz.

# Sequential model oluşturuluyor. Bu model, katmanları sıralı bir şekilde bir araya getirir.
model = tf.keras.Sequential([
    # Flatten katmanı, 28x28 boyutundaki giriş verisini tek boyutlu bir vektöre düzleştirir.
    tf.keras.layers.Flatten(input_shape=(28, 28)), 

    # Dense katmanı, 128 nöron içerir ve 'relu' aktivasyon fonksiyonu kullanır.
    tf.keras.layers.Dense(128, activation='relu'), 

    # Dense katmanı, 10 nöron içerir. Bu katman, modelin çıkışını oluşturur.
    # Bu örnekte, 10 nöron, çok sınıflı bir sınıflandırma problemi için kullanılır.
    tf.keras.layers.Dense(10) 
])

# Modeli derleme işlemi
model.compile(
    # Optimizasyon algoritması belirlenir. 'adam' optimizasyonu, adaptif öğrenme hızlarıyla birlikte çok kullanılan bir optimizasyon algoritmasıdır.
    optimizer='adam',

    # Kayıp fonksiyonu belirlenir. Bu örnekte, Sparse Categorical Crossentropy kullanılır.
    # Sparse Categorical Crossentropy, çok sınıflı sınıflandırma problemleri için yaygın olarak kullanılan bir kayıp fonksiyonudur.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    # Modelin değerlendirilmesi için kullanılacak metrikler belirlenir. Bu örnekte, 'accuracy' (doğruluk) metriği kullanılır.
    metrics=['accuracy']
)


# Modelin eğiltime işlemi bu kısımda gerçekleştirilir.
model.fit(train_images, train_labels, epochs =10)

# Modele verilen test verileri sayesinde başarı oranı ortaya çıkartılır.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)

# Yeni bir model oluşturuluyor: 'probability_model'
# Bu model, önceki modelin çıkışını alacak ve bu çıkışa softmax aktivasyonunu uygulayacaktır.
probability_model = tf.keras.Sequential([
    model,  # Önceki model (giriş verisi ve diğer katmanlar)
    tf.keras.layers.Softmax()  # Softmax aktivasyonu, çıkışı olasılık dağılımına dönüştürür.
])

# Test görüntüleri üzerinde tahminlerde bulunur.
predictions = probability_model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
