"""
Rancangan program untuk Deep Belief Network.
"""

from PIL import Image
import natsort
import numpy as np
import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification

# daftar label
df = pd.read_csv('trainLabels.csv')
# daftar citra yang diurutkan
path = 'citra/'
dir = natsort.natsorted(listdir(path))
# hapus citra yang error
available_image = set(dir).intersection(set(df.image))
# list citra
images = []
for img in dir:
        # buka citra
        im = Image.open(path + img)
        # ubah ke dalam array
        data = np.array(im)
        # ubah ukuran citra menjadi (3 * 256 * 256)
        flattened = data.flatten()
        # tambahkan citra ke list
        images.append(flattened)
# ubah ukuran citra menjadi (jumlah citra, 3 * 256 * 256)
# masukan model
X = np.array(images)        
# keluaran model
Y = df.level
# split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# object pelatihan model
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
# latih model
classifier.fit(X_train, y_train)

# simpan model
#classifier.save('model.pkl')
# load model
#classifier = SupervisedDBNClassification.load('model.pkl')

# prediksi
y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(y_test, y_pred))