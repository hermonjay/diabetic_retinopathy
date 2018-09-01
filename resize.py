"""
Langkah pertama dalam pre-processing.
Mengubah ukuran citra.
"""

from os import listdir
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# hasil ukuran citra yang diinginkan
shape = 256
# alamat citra
path = 'train/'

def resize():
    # daftar citra dalam list
    dir = listdir(path)
    # loop di setiap citra dalam list
    for item in dir:
        # buka citra
        img = Image.open(path + item)
        # ubah ukuran citra
        img = ImageOps.fit(img, (shape, shape))
        # simpan citra
        img.save(path + item)
    print('Done resizing!') 

resize()