# Imports here
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import glob
import random
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.preprocessing import LabelEncoder
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()  # Corrected line
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # Adjusted this line
        self.fc2 = nn.Linear(512, 102)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
#checkpoint = torch.load('transfer_learning_model_final.pth', map_location=device)
#print(type(checkpoint))


# Define the image transformations for the train and validation and test sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('splitted_dataset/train', transform=train_transforms)
valid_data = datasets.ImageFolder('splitted_dataset/valid', transform=valid_transforms)
test_data = datasets.ImageFolder('splitted_dataset/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Get class names from the 'class_to_idx' attribute of the dataset
class_names = list(train_data.class_to_idx.keys())

# Initialize and fit the LabelEncoder with the class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Save the fitted LabelEncoder to a file for later use during inference
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Load the model (assumes you're retraining from scratch)
model = CNNModel()
model.to(device)
# Define the loss and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# List to store epoch results
results = []

# Training Loop
epochs = 100
for epoch in range(epochs):
    running_loss = 0
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")
       

# Writing results to a CSV file
with open('training_results.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train Loss','Validation Loss', 'Validation Accuracy'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Save the model
torch.save(model.state_dict(), 'model.pth')

