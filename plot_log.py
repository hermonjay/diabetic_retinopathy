"""
Plot proses pelatihan
"""

import matplotlib.pyplot as plt
import pandas as pd

# load csv, argumen = alamat file (hasil callback CSVLogger)
log = pd.read_csv('assets/log_cnn02_28-8.csv')

# ukuran figure
plt.figure(figsize=(10,5))
# plot, argumen = (x, y)
plt.plot(log['epoch'], log['loss'])
# label dan judul
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss Graph')

plt.show()