import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import LabelEncoder
import csv
from tqdm import tqdm

import torch.nn as nn
import torchvision.models as models

import torch.nn as nn


import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjusted to match the flattened size from the error message
        self.fc1 = nn.Linear(700928, 512)

        self.fc2 = nn.Linear(512, 102)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
# Define image transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(299),  # Inception V3 expects 299x299 input
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # New augmentation
    transforms.ColorJitter(),  # New augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

valid_transforms = transforms.Compose([
    transforms.Resize(320),  # Slightly larger than 299 for cropping
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder('dataset/splitted_dataset/train', transform=train_transforms)
valid_data = datasets.ImageFolder('dataset/splitted_dataset/valid', transform=valid_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
validloader = DataLoader(valid_data, batch_size=32)


# Get class names from the 'class_to_idx' attribute of the dataset
class_names = list(train_data.class_to_idx.keys())

# Initialize and fit the LabelEncoder with the class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Save the fitted LabelEncoder to a file for later use during inference
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)


# Initialize Inception V3 with pre-trained weights
model = CNNModel().to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

# Early stopping parameters
early_stopping_patience = 10
min_val_loss = float('inf')
no_improvement_epochs = 0

# List to store epoch results
results = []

# Training loop
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    model.train()  # Set the model to training mode
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        # We need to access the main output for computing the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100*correct_train / total_train

    # Validation loop...
    valid_loss = 0
    accuracy = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Again, we access the main output for the loss
            valid_loss += criterion(outputs, labels).item()
            
            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels.data).item()
    
    accuracy = 100*accuracy / len(validloader.dataset)
     # Print out the losses and accuracy
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Train accuracy: {train_accuracy:.3f}.. "
          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {accuracy:.3f}")

    # Append the results of this epoch to the results list
    results.append({
        'Epoch': epoch + 1,
        'Train Loss': running_loss / len(trainloader),
        'Train Accuracy': train_accuracy,
        'Validation Loss': valid_loss / len(validloader),
        'Validation Accuracy': accuracy
    })

    # Early stopping check
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        no_improvement_epochs = 0
        # Save the model only when validation loss decreases
        torch.save(model.state_dict(), 'Models/base.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Scheduler step (after validation)
    scheduler.step(valid_loss)  #

# Save the model if not done in the early stopping
if no_improvement_epochs < early_stopping_patience:
    torch.save(model.state_dict(), 'Models/base.pth')

# Writing results to a CSV file
with open('Results/base_results.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train Loss', 'Train Accuracy','Validation Loss', 'Validation Accuracy'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Training results saved to training_results.csv")