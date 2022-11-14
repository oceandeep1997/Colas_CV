import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import ipdb
import sys
from model_building.model_creation import *
from model_building.multi_output_model import *
import config
import pandas as pd
# from tensorflow.keras.utils import load_img
# from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split

# dataset_path = "/home/jovyan/hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/"
# dataset_path = os.path.join("Data","images_train_subset")
if __name__ == "__main__":
    train_labels = pd.read_csv(os.path.join(config.dataset_path,"labels_train.csv"))
    df_train, df_val, df_test = np.split(
        train_labels.sample(frac=1, random_state=42),
        [int(0.8 * len(train_labels)), int(0.9 * len(train_labels))],
    )
    class_proportions = np.flip(df_train.iloc[:,1:].apply(pd.Series.value_counts).T.values,axis=1)
    class_weights = 1 / (class_proportions / class_proportions.sum(axis=1).reshape((-1,1)))
    class_weights = torch.tensor(class_weights)
    # train_transform = create_image_transform()
    # train_dataset = Colas_Dataset(
    #     df_train, os.path.join(dataset_path, "train"), transform=train_transform
    # )
    # val_dataset = Colas_Dataset(
    #     df_val, os.path.join(dataset_path, "train"), transform=train_transform
    # )
    colas_model = multi_output_model_colas(df_train.shape[1]-1)
    colas_model.train(
        df_train=df_train,
        df_val=df_val,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        dataset_path=config.dataset_path
    ) 
    df_test = pd.read_csv(os.path.join(config.dataset_path,"template_test.csv"))
    try:
        predictions = colas_model.predict(df_test, config.batch_size)
    except:
        ipdb.set_trace()
    ipdb.set_trace()