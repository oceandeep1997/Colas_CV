import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import os
import copy
import tqdm
import ipdb
import sys

def create_image_transform(random_size_crop:int = 224):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(random_size_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class Colas_Dataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name = self.data[index][0]
        labels = self.data[index][1:]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, labels

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

data_dir = '/data/train'


class colas_model:

    def __init__(self,model:multi_output_model, number_outputs:int=4) -> None:
        self.model = model 
        self.number_outputs = number_outputs
        self.is_model_trained = False
 
    def train(self, train_data, val_data, optimizer, batch_size , num_epochs=25):
        train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
        val_dataloader = DataLoader(val_data,batch_size=batch_size,shuffle=True)

        global_criterion = nn.BCELoss()
        for i in range(1,self.number_outputs+1):
            globals[f'criterion_output_{i}'] = nn.BCELoss()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            self.model = self.model.cuda()
            globals[f'criterion_output_{i}'] = globals[f'criterion_output_{i}'].cuda()        

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            
            total_loss_train = 0
            total_acc_train = 0

            for train_input, train_labels in tqdm(train_dataloader):
                train_labels = train_labels.to(device)
                train_input = train_input.to(device)
                outputs = self.model(train_input)
                batch_loss = 0
                for i in self.number_outputs:
                    loss_output = global_criterion(outputs, train_labels)
                    globals[f"loss_output_{i}"] = globals[f'criterion_output_{i}'](outputs[i],train_labels[i]) 
                    batch_loss += globals[f"loss_output_{i}"]


                total_loss_train += batch_loss

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()

        self.is_model_trained = True

    def predict_proba(self,test_data, batch_size, ):
        # if not self.is_model_trained:
        #     raise AttributeError(
        #         "the model has not yet been trained, train it first before predictions"
        #     )
        #     sys.exit()
        
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
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
            # print('-' * 10)

            # # Each epoch has a training and validation phase
            # for phase in ['train', 'val']:
            #     if phase == 'train':
            #         self.model.train()  # Set model to training mode
            #     else:
            #         self.model.eval()   # Set model to evaluate mode

            #     running_loss = 0.0
            #     running_corrects = 0

            #     # Iterate over data.
            #     for inputs, labels in dataloaders[phase]:
            #         inputs = inputs.to(device)
            #         labels = labels.to(device)

            #         # zero the parameter gradients
            #         optimizer.zero_grad()

            #         # forward
            #         # track history if only in train
            #         with torch.set_grad_enabled(phase == 'train'):
            #             outputs = model(inputs)
            #             _, preds = torch.max(outputs, 1)
            #             loss = criterion(outputs, labels)

            #             # backward + optimize only if in training phase
            #             if phase == 'train':
            #                 loss.backward()
            #                 optimizer.step()

            #         # statistics
            #         running_loss += loss.item() * inputs.size(0)
            #         running_corrects += torch.sum(preds == labels.data)
            #     if phase == 'train':
            #         scheduler.step()

            #     epoch_loss = running_loss / dataset_sizes[phase]
            #     epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            #     # deep copy the model
            #     if phase == 'val' and epoch_acc > best_acc:
            #         best_acc = epoch_acc
            #         best_model_wts = copy.deepcopy(model.state_dict())

            # print()

        # time_elapsed = time.time() - since
        # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Acc: {best_acc:4f}')

        # # load best model weights
        # model.load_state_dict(best_model_wts)
        # return model