"""
Evaluasi model.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator  

# load model
model = load_model('assets/model_cnn02_28-8.h5')
# alamat data uji
test_path = 'prepared_folder/test/'
# buat object generator untuk data uji
# normalisasi nilai dengan membagi setiap piksel dengan 255
test_datagen = ImageDataGenerator(rescale=1. / 255)
# gunakan method untuk menerapkan generator pada alamat citra
test_generator = test_datagen.flow_from_directory(
        # alamat citra
        train_data_path,  
        # ukuran citra
        target_size=(256, 256),  
        # batch size (disesuaikan dengan kemampuan GPU)
        batch_size=16,
        # tipe kelas = kategorikal
        class_mode='categorical', 
        # acak citra
        shuffle=True)  

# evaluasi model
loss, acc = model.evaluate_generator(test_generator)
# akurasi model
acc = acc*100
print('Loss     : %.3f' %loss)
print('Accuracy : %.3f' %acc, '%')
# daftar kelas data uji
y_test = test_generator.classes
# prediksi pada data uji
y_pred_gen = model.predict_generator(test_generator)
y_pred = np.argmax(y_pred_gen, axis=1)

# akurasi model
akurasi = accuracy_score(y_test, y_pred) * 100
# daftar nilai precission, recall, dan F1 score
report = classification_report(y_test, y_pred)
print(report)
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print()
print(cm)