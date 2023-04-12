import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from tqdm import tqdm

#define the model
def get_model():

    input = Input(shape=(None, None, 1))
    #n_inp = input/255
    x = Conv2D(32, 3, activation='relu', padding='same')(input)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation=None, padding='same')(x)
    x = Activation('tanh') (x)
    x = x * 127.5 + 127.5

    model = Model([input], x)
    model.summary()
    return model

def get_data():
    x = []
    y = []
    for img_dir in tqdm(glob.glob('C:\\Users\\zifen\\Desktop\\4TN4\\Projects\\DIV2K_train_HR\\000*.png')):
        img = cv2.imread(img_dir)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        # better way: "in" should be the downsampled y by your algorithm!
        # better: only pick patch at each epoch! no resize the whole image
        y_out = cv2.resize(y_channel, (128, 128), interpolation=cv2.INTER_AREA)
        y_in = cv2.resize(y_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        x.append(y_in)
        y.append(y_out)

    x = np.array(x)
    y = np.array(y)

    return x, y

model = get_model()
# second step we need a dataloader
x, y = get_data()
print(x.shape, y.shape)

plt.subplot(211)
plt.imshow(x[0], cmap = 'gray')
plt.subplot(212)
plt.imshow(y[0], cmap = 'gray')
plt.show()

from sklearn.model_selection import train_test_split
import tensorflow as tf
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42)

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-04) # can be tuned as hyperparamter
loss = 'mse' # can be other losses!
model.compile(loss='mse', optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint('model/model2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_freq='epoch')

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

batch_size = 4
epochs = 10 # 100 maybe!
# can get data loader as the input!
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[save_model_callback, tbCallBack])


# inference part! for the new image!
model = load_model('model/model2.h5')
img = cv2.imread('C:\\Users\\Basir\\Documents\\ECE\\Final_year\\4TN4\\Proj2\\CNN\\dataset_images\\DIV2K_valid_HR\\0801.png')
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y_channel = img_ycrcb[:, :, 0]

y_in = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

y = cv2.resize(y_channel, (64, 64), interpolation=cv2.INTER_AREA)

y = np.expand_dims(y, axis=0)

# if you have preprocessing you may want to apply those here!
y_upsampled = model.predict(y)

plt.subplot(211)
plt.imshow(y[0], cmap='gray')
plt.subplot(212)
plt.imshow(y_upsampled[0], cmap='gray')
plt.show()