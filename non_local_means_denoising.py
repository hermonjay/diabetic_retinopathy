"""
Langkah ketiga dalam pre-processing.
Non-local Means Denoising.
"""

from os import listdir
import cv2

# alamat citra    
path = 'train/'

def nlmd():    
    # daftar citra dalam list
    dir = listdir(path)
    # loop di setiap citra dalam list
    for img in dir:
        # buka citra
        bgr = cv2.imread(path + img)
        # gunakan fungsi NLMD
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 19)
        # simpan citra
        cv2.imwrite(path + img, bgr)    
    print('Done applying NLMD!')

nlmd()