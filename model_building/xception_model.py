from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

#Functions


from keras import backend as K

def oversampler(X, y):    
    X = list(X)
    counter = int(y.mean() * len(y))
    angles = [90, 180, 270]
    i = 0
    angle = 90
    while counter / len(y) < 0.5:
        for i in range(len(y)):
            if y[i] == 1:
                # get dims, find center
                image = X[i]
                (h, w) = image.shape[:2]
                (cX, cY) = (w // 2, h // 2)

                # grab the rotation matrix (applying the negative of the
                # angle to rotate clockwise), then grab the sine and cosine
                # (i.e., the rotation components of the matrix)
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY

                # perform the actual rotation and return the image
                image = cv2.warpAffine(image, M, (nW, nH), False)

                X.append(image)
                y = np.append(y, y[i])
                counter += 1
            if counter / len(y) >= 0.5:
                break

        i += 1
        angle = angles[i%3]
    X = np.array(X)
    return X, y

class model_xception():
    def __init__(self):
        base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                            include_top=False,
                                                            input_shape = (224, 224, 3))

         # Flatten the output layer to 1 dimension
        x = layers.Flatten()(base_model.output)
        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = layers.Dense(512, activation='relu')(x)
        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)
        # Add a final sigmoid layer with 1 node for classification output
        x = layers.Dense(1, activation='sigmoid')(x)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model = tf.keras.models.Model(base_model.input, x)
        # compile the model
        self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',
                metrics=[ f1_m,precision_m, 'acc', recall_m])

    def train(self, X_train, y_train,validation_data, epochs, batch_size):
      
        self.model.fit(X_train, y_train, validation_data = validation_data, epochs = epochs, batch_size = batch_size)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
    
    def predict(self, df, y_test):
        test_pred = self.model.predict(df)
        test_pred_1 = np.round(test_pred)
        prior_adjusted_test_pred = np.round(test_pred * y_test.mean() / test_pred_1.mean())
        prior_adjusted_test_pred = prior_adjusted_test_pred.astype(int)        
        return prior_adjusted_test_pred
