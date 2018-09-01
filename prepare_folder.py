"""
Atur folder agar sesuai format ImageDataGenerator keras.
"""

import pandas as pd
import natsort
import os
from shutil import copy2

# daftar label
df = pd.read_csv('trainLabels.csv')
# daftar citra yang diurutkan
dir = natsort.natsorted(os.listdir('images/'))
# hapus citra yang error
available_image = set(dir).intersection(set(df.image))

# iterasi untuk meng-copy citra sesuai kelasnya
for i in range(len(dir)):
    if 0 == int(df.level[i]):
        copy2('train/' + str(dir[i]), 'prepared_folder/0/')
    elif 1 == int(df.level[i]):
        copy2('train/' + str(dir[i]), 'prepared_folder/1/')
    elif 2 == int(df.level[i]):
        copy2('train/' + str(dir[i]), 'prepared_folder/2/')
    elif 3 == int(df.level[i]):
        copy2('train/' + str(dir[i]), 'prepared_folder/3/')
    elif 4 == int(df.level[i]):
        copy2('train/' + str(dir[i]), 'prepared_folder/4/')
    else:
        pass