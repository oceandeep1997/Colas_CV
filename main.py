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
import config
import pandas as pd
# from tensorflow.keras.utils import load_img
# from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split

dataset_path = "/home/jovyan/hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/"
if __name__ == "__main__":
    train_labels = pd.read_csv(dataset_path + "labels_train.csv")
    df_train, df_val, df_test = np.split(
        train_labels.sample(frac=1, random_state=42),
        [int(0.8 * len(train_labels)), int(0.9 * len(train_labels))],
    )
    train_transform = create_image_transform()
    train_dataset = Colas_Dataset(
        df_train, os.path.join(dataset_path, "train"), transform=train_transform
    )
    val_dataset = Colas_Dataset(
        df_val, os.path.join(dataset_path, "train"), transform=train_transform
    )
    model_cnn = multi_output_model()
    colas_model = colas_model(model_cnn, 5)
    colas_model.train(
        train_data=train_dataset,
        val_data=val_dataset,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )
