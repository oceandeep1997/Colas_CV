from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random as rand
#!pip install opencv-python
import cv2
import math

import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from xception_model import *
batchsize=32


def main():
    dataset_path = "../hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/"
    test_names= pd.read_csv(dataset_path + "template_test.csv")
    train_labels= pd.read_csv(dataset_path + "labels_train.csv")
    
    ## prepare the train and test dataset
    train_image = []
    for i in tqdm(range(train_labels.shape[0])):
        img = load_img(dataset_path + "train/" + train_labels["filename"][i], target_size=(224,224,3))
        img = img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)
    test_image = []
    for i in tqdm(range(test_names.shape[0])):
        img = load_img(dataset_path + "test/" + test_names["filename"][i], target_size=(224,224,3))
        img = img_to_array(img)
        img = img/255
        test_image.append(img)
    X_test = np.array(test_image)
    
    y = np.array(train_labels.drop(["filename"], axis=1))
    y_fissure = np.array(train_labels["FISSURE"])
    y_reparation = np.array(train_labels["REPARATION"])
    y_longi = np.array(train_labels["FISSURE LONGITUDINALE"])
    y_faience = np.array(train_labels["FAÏENCAGE"])
    y_med = np.array(train_labels["MISE EN DALLE"])
    
    #### FISSURE
    X_fissure_train, X_fissure_test, y_fissure_train, y_fissure_test = train_test_split(X, y_fissure, test_size=0.2)
    X_fissure_train, y_fissure_train = oversampler(X_fissure_train, y_fissure_train) 
    
    #### REPARATION
    X_reparation_train, X_reparation_test, y_reparation_train, y_reparation_test = train_test_split(X, y_reparation, test_size=0.2)
    X_reparation_train, y_reparation_train = oversampler(X_reparation_train, y_reparation_train) 
    #### FISSURE LONGITUDINALE
    X_longi_train, X_longi_test, y_longi_train, y_longi_test = train_test_split(X, y_longi, test_size=0.2)
    X_longi_train, y_longi_train = oversampler(X_longi_train, y_longi_train) 
    
    #### FAIENÇAGE
    X_faience_train, X_faience_test, y_faience_train, y_faience_test = train_test_split(X, y_faience, test_size=0.2)
    X_faience_train, y_faience_train = oversampler(X_faience_train, y_faience_train) 
    
    #### MISE EN DALLE
    X_med_train, X_med_test, y_med_train, y_med_test = train_test_split(X, y_med, test_size=0.2)
    X_med_train, y_med_train = oversampler(X_med_train, y_med_train) 

    
    # train the model and make the prediction one by one
    model = model_xception()
    model.train(X_fissure_train, y_fissure_train, validation_data = (X_fissure_test, y_fissure_test), epochs = 20, batch_size = 32)
    prior_adjusted_fissure_pred = model.predict(X_test,y_fissure_test)
    model.train(X_reparation_train, y_reparation_train, validation_data = (X_reparation_test, y_reparation_test), epochs = 20, batch_size = 32)
    prior_adjusted_reparation_pred = model.predict(X_test,y_reparation_test)
    
    model.train(X_longi_train, y_longi_train, validation_data = (X_longi_test, y_longi_test), epochs = 20, batch_size = 32)
    prior_adjusted_longi_pred = model.predict(X_test,y_longi_test)
    
    model.train(X_faience_train, y_faience_train, validation_data = (X_faience_test, y_faience_test), epochs = 20, batch_size = 32)
    prior_adjusted_faience_pred = model.predict(X_test,y_faience_test)
    
    model.train(X_med_train, y_med_train, validation_data = (X_med_test, y_med_test), epochs = 20, batch_size = 32)
    prior_adjusted_med_pred = model.predict(X_test,y_med_test)
    
    #finalize the file and save it
    test_labels = test_names.copy()
    test_labels["FISSURE"] = prior_adjusted_fissure_pred
    test_labels["REPARATION"] = prior_adjusted_reparation_pred
    test_labels["FISSURE LONGITUDINALE"] = prior_adjusted_longi_pred
    test_labels["FAÏENCAGE"] = prior_adjusted_faience_pred
    test_labels["MISE EN DALLE"] = prior_adjusted_med_pred
    test_labels.to_csv("xception2.csv", index=False)

    

    
if __name__ == "__main__":
    main()
