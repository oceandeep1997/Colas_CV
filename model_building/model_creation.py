import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import os
import tqdm
import ipdb
import sys
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import config


def create_image_transform(random_size_crop: int = 256) -> transforms.Compose:
    """
    Creates image transformation pipeline used in CNN model later on
    args:
        - random_size_crop: int>0, size to reshape the picture for model compatibility

    returns:
        - trans_transforms : transforms.Compose pipeline for images
    """
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(random_size_crop),
            # transforms.CenterCrop(random_size_crop),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transforms


def compute_accuracy_values(y_true: torch.tensor, y_predicted: torch.tensor):
    """
    computes accuracy sum for a tensor batch. Works for multi-outputs models

    args:
        - y_true: true_labels
        - y_predicted: predicted labels by model

    returns:
        - acc: float, total accuracy
    """
    y_classes = 1 * (y_predicted > 0.5)
    try:
        acc = (1 * (y_true == y_classes)).detach().cpu().numpy()
    except:
        ipdb.set_trace()
    return acc


class Colas_Dataset(Dataset):
    def __init__(self, data: pd.DataFrame, path: str, transform=None) -> None:
        """
        Colas Dataset class : friendly PyTorch Class for the Colas Computer vsion project. Class works both for single output models or multi-outputs models

        args:
            - data: dataframe containing name of each picture and associated label(s)
            - path: data path
            - transform: torch transforms.Compose pipeline, by default non-used
        """
        super(Colas_Dataset,self).__init__()
        self.data = data.values
        self.labels = data.iloc[:, 1].values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """ "
        gets item of data based on passed index
        args:
            -index : int, index of next input in dataframe
        returns:
            - image, labels : plt image and associated label(s)
        """

        img_name = self.data[index][0]
        labels = np.array(self.data[index][1:], dtype="float32")
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        # ipdb.set_trace()
        return image, labels

    def classes_imbalance_sampler(self) -> WeightedRandomSampler:
        """
        returns a WeightedRandomSampler to handle unbalanced classes
        args:
            -none
        returns:
            -sampler : WeightedRandomSampler which will draw data in dataset based on their overall repartition
        """
        targets = self.labels
        try:
            class_sample_count = np.array(
                [len(np.where(targets == t)[0]) for t in np.arange(0, max(targets) + 1)]
            )
        except:
            ipdb.set_trace()
        weight = 1.0 / (class_sample_count + 0.1)
        # ipdb.set_trace()
        weights = list()
        for t in targets:
            try:
                weights.append(weight[t])
            except:
                ipdb.set_trace()
        samples_weight = np.array(weights)
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler


class single_output_model_vgg(nn.Module):
    def __init__(
        self,
        neuron_mid_layer: int = 40,
        dropout: float = 0.4,
    ) -> None:
        """
        cnn model that fine-tunes vgg16 model gathered with pytorch

        args:
            -neuron_mid_layer : int>0, number of neurons in the penultimate layer
            -dropout : float belonging to (0,1) dropout rate
        """
        super(single_output_model_vgg, self).__init__()
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        number_features = self.model.classifier[6].out_features
        # self.model_resnet.fc = nn.Identity()
        self.first_layer = nn.Linear(number_features, int(number_features / 2))
        self.first_relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(int(number_features / 2))
        self.middle_layer = nn.Linear(int(number_features / 2), neuron_mid_layer)
        self.relu_middle_layer = nn.ReLU()
        self.batch_norm_2 = nn.BatchNorm1d(neuron_mid_layer)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(neuron_mid_layer, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.first_layer(x)
        x = self.first_relu(x)
        x = self.dropout(x)
        x = self.batch_norm_1(x)
        x = self.middle_layer(x)
        x = self.relu_middle_layer(x)
        x = self.dropout(x)
        x = self.batch_norm_2(x)
        x = self.fc1(x)
        x = self.final_sigmoid(x)
        return x


class single_output_model(nn.Module):
    def __init__(
        self,
        neuron_mid_layer: int = 40,
        dropout: float = 0.4,
    ) -> None:
        super(single_output_model, self).__init__()
        self.model_resnet = models.resnet18(pretrained=True)
        number_features = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.middle_layer = nn.Linear(number_features, neuron_mid_layer)
        self.relu_middle_layer = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(neuron_mid_layer, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model_resnet(x)
        x = self.middle_layer(x)
        x = self.relu_middle_layer(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.final_sigmoid(x)
        return x


data_dir = "/data/train"


class colas_model_single_output:
    def __init__(self, model: single_output_model_vgg) -> None:
        """
        class used to train a colas cnn model with one output. In the end, we'll train as many models like that as this model
        args:
            - model : single_output_model_vgg, cnn model used and trained
        """
        self.model = model
        self.is_model_trained = False

    def train(
        self,
        train_data: Colas_Dataset,
        val_data: Colas_Dataset,
        learning_rate: float,
        use_samplers: bool = True,
        batch_size: int = 64,
        num_epochs=config.number_epochs,
    ):
        """
        trains declared model during initialization

        args:
            - train_data : Colas_Dataset, torch-friendly train dataset
            - validaiton_data : Colas_Dataset, torch-friendly validation dataset
            - learning_rate : float>0, learning rate used in training
            - use_samples: bool, wether to use WeightedRandomSamplers for image selection during training (used it if unbalanced classes)
            - batch_size: int, batch_size
            - num_epochs: int, maximal number of epochs for training

        returns:
            - f1_score_val : float, f1-score for validation set
        """
        print(batch_size)
        if use_samplers:
            train_sampler = train_data.classes_imbalance_sampler()
            val_sampler = val_data.classes_imbalance_sampler()
            train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, sampler=train_sampler
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=batch_size, sampler=val_sampler
            )
        else:
            train_dataloader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True
            )
            val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        global_criterion = nn.BCELoss()
        # for i in range(1,self.number_outputs+1):
        #     globals[f'criterion_output_{i}'] = nn.BCELoss()
        # adding nothing weird stuff

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            self.model = self.model.cuda()
            global_criterion = global_criterion.cuda()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")

            total_loss_train = 0
            total_acc_train = []
            list_outputs = []
            list_train_labels = []
            for train_input, train_labels in tqdm.tqdm(train_dataloader):
                train_labels = train_labels.to(device)
                train_input = train_input.to(device)
                
                try:
                    outputs = self.model(train_input)
                    batch_loss = global_criterion(outputs, train_labels)
                except:
                    ipdb.set_trace()
                total_loss_train += batch_loss

                list_outputs += [
                    element[0]
                    for element in (1 * (outputs > 0.5)).detach().cpu().numpy().tolist()
                ]
                list_train_labels += [
                    element[0]
                    for element in train_labels.detach().cpu().numpy().tolist()
                ]

                accuracy_train = compute_accuracy_values(train_labels, outputs)
                total_acc_train += list(accuracy_train)

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = []
            total_loss_val = 0
            list_outputs_val = []
            list_val_labels = []
            with torch.no_grad():
                for val_input, val_labels in val_dataloader:

                    val_labels = val_labels.to(device)
                    val_input = val_input.to(device)

                    outputs = self.model(val_input)

                    batch_loss = global_criterion(outputs, val_labels)
                    total_loss_val += batch_loss.item()

                    accuracy_batch = compute_accuracy_values(val_labels, outputs)
                    total_acc_val += list(accuracy_batch)
                    list_outputs_val += [
                        element[0]
                        for element in (1 * (outputs > 0.5))
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    ]
                    list_val_labels += [
                        element[0]
                        for element in val_labels.detach().cpu().numpy().tolist()
                    ]
            try:
                f1_score_val = f1_score(list_val_labels, list_outputs_val)
                f1_score_train = f1_score(list_train_labels, list_outputs)
                recall_score_val = recall_score(list_val_labels, list_outputs_val)
                precision_score_val = precision_score(list_val_labels, list_outputs_val)
                print(
                    f"f1_score_train = {f1_score_train : .3f} and f1_score_val = {f1_score_val: .3f}"
                )
                print(
                    f"recall validation {recall_score_val:.3f}, precision_score val : {precision_score_val:.3f}"
                )
            except:
                print("mistake here")
                ipdb.set_trace()

        self.is_model_trained = True
        return f1_score_val

    def predict_proba(
        self,
        test_data: Colas_Dataset,
        batch_size: int = 32,
    ):
        """
        returns predicted class-belonging probabilities for inputs

        args:
            - test_data: Colas_Dataset, torch-friendly test dataset
            - batch_size: int>0, batch_size

        returns:
            - predictions : list containing predicted probabilities
        """
        if not self.is_model_trained:
            raise AttributeError(
                "the model has not yet been trained, train it first before predictions"
            )
            sys.exit()

        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()

        predictions = []
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                test_input = test_input.to(device)

                output = self.model(test_input)
                predictions.append(output.detach().cpu().numpy())

        return predictions

    def predict(
        self,
        test_data,
        batch_size,
    ):
        """
        returns predicted labels for the inputs

        args:
            - test_data: Colas_Dataset, torch-friendly test dataset
            - batch_size: int>0, batch_size

        returns:
            - predictions : list containing predicted labels
        """
        if not self.is_model_trained:
            raise AttributeError(
                "the model has not yet been trained, train it first before predictions"
            )
            sys.exit()

        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()
        
        
        predictions = []
        with torch.no_grad():
            
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                test_input = test_input.to(device)

                outputs = self.model(test_input)
                predictions.append((1 * (outputs > 0.5)).detach().cpu().numpy())

        
        return predictions
