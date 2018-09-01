"""
Langkah kedua dalam pre-processing.
Contrast-Limited Adaptive Histogram Equalization pada channel LAB (L).
"""

from os import listdir
import cv2

# alamat citra    
path = 'train/'

def clahe():        
    # daftar citra dalam list
    dir = listdir(path)      
    # loop di setiap citra dalam list  
    for img in dir:
        # buka citra
        bgr = cv2.imread(path + img)
        # konversi channel citra ke LAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        # split channel
        lab_planes = cv2.split(lab)
        # object CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        # panggil method untuk menerapkan CLAHE ke channel L (indeks 0)
        lab_planes[0] = clahe.apply(lab_planes[0])
        # gabungkan channel LAB
        lab = cv2.merge(lab_planes)
        # konversi channel citra ke BGR
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # simpan citra
        cv2.imwrite(path + img, bgr)    
    print('Done applying CLAHE!')

clahe()