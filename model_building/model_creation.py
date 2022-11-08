import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


class multi_output_model(nn.Module):

    def __init__(self,neuron_mid_layer:int=40,dropout:float=0.4) -> None:
        super(multi_output_model,self).__init__()
        self.model_resnet = models.resnet18(pretrained=True)
        number_features = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.middle_layer = nn.Linear(number_features, neuron_mid_layer)
        self.relu_middle_layer = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(neuron_mid_layer, 1)
        self.fc2 = nn.Linear(neuron_mid_layer,1)
        self.fc3  = nn.Linear(neuron_mid_layer,1)
        self.fc4  = nn.Linear(neuron_mid_layer,1)

    def forward(self,x):
        x = self.model_resnet(x)
        x = self.middle_layer(x)
        x = self.relu_middle_layer(x)
        x = self.dropout(x)
        output_1 = self.fc1(x)
        output_2 = self.fc2(x)
        output_3 = self.fc3(x)
        output_4 = self.fc4(x)

        return output_1, output_2, output_3, output_4

class colas_model:

    def __init__(self,model, dataloaders) -> None:
        self.model = model
        self.dataloaders = dataloaders
        pass

    def train(self, criterion, optimizer, scheduler, num_epochs=25):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model