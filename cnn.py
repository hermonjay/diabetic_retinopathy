"""
Pembuatan dan pelatihan model CNN.
"""

import keras.applications
from keras.models import Model
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator  
from keras import optimizers, callbacks

# generator untuk data latih
def create_train_generator(train_data_path):  
    # buat object generator untuk data latih
    # normalisasi nilai dengan membagi setiap piksel dengan 255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # gunakan method untuk menerapkan generator pada alamat citra
    train_generator = train_datagen.flow_from_directory(
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
    return train_generator  

# generator untuk data validasi
def create_val_generator(train_data_path):  
    # buat object generator untuk data validasi
    # normalisasi nilai dengan membagi setiap piksel dengan 255
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # gunakan method untuk menerapkan generator pada alamat citra
    val_generator = val_datagen.flow_from_directory(  
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
    return val_generator  

# pembuatan model
def create_model_vgg(num_class):
    # panggil model (tanpa fully connected layers)
    vgg_model = keras.applications.vgg16.VGG16(
        include_top=False, weights='imagenet', pooling='avg', input_shape=(256, 256, 3))
    
    # freeze enam layer pertama
    for layer in vgg_model.layers[:6]:
        layer.trainable = False        

    # lanjutkan model setelah lapisan konvolusi
    x = vgg_model.output
    # tambahkan dropout
    x = Dropout(0.5)(x)
    # tambahkan fully connected layers dengan jumlah unit 512 dan fungsi aktivasi ReLU
    x = Dense(512, activation="relu")(x)
    # tambahkan dropout
    x = Dropout(0.5)(x)
    # tambahkan fully connected layers dengan jumlah unit 512 dan fungsi aktivasi ReLU
    x = Dense(512, activation="relu")(x)
    # tambahkan lapisan untuk prediksi dengan fungsi aktivasi softmax
    predictions = Dense(num_class, activation="softmax")(x)    
    # model keseluruhan
    model_final = Model(inputs=vgg_model.input, outputs=predictions)    
    return model_final

# buat model, argumen = jumlah kelas
model = create_model_vgg(5)
# buat generator data latih, argumen = alamat data latih
train_generator = create_train_generator('prepared_folder/train/')
# buat generatordata validasi, argumen = alamat data validasi
val_generator = create_val_generator('prepared_folder/val/')
# buat log pelatihan, argumen = nama file
csv_logger = callbacks.CSVLogger('log_cnn02_28-8.csv')
# learning rate, hyperparameter [1e-03, 1e-04, 1e-05]
learning_rate = 0.0001
# jumlah iterasi
epoch = 100
# metode backpropagation, https://keras.io/optimizers/
# argumen = learning rate, learning rate decay (optional)
opt = optimizers.SGD(lr=learning_rate, decay=learning_rate/epoch)

# compile model
model.compile(optimizer=opt,
              # hitung loss, https://keras.io/losses/
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# latih model
history = model.fit_generator(
    # data latih
    train_generator,        
    # jumlah iterasi
    epochs=epoch,
    # data validasi
    validation_data=val_generator,
    # callback, https://keras.io/callbacks/
    callbacks=[csv_logger])    

# simpan model, argumen = nama file dalam h5
model.save('model_cnn02_28-8.h5')