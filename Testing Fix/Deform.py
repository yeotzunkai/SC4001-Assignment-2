import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torchvision
from tqdm import tqdm  # Standard import for scripts and applications
import pandas as pd
class DeformableCNNModel(nn.Module):
    def __init__(self):
        super(DeformableCNNModel, self).__init__()

        # Convolutional weights
        self.weight1 = nn.Parameter(torch.Tensor(32, 3, 3, 3))
        self.weight2 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.weight3 = nn.Parameter(torch.Tensor(128, 64, 3, 3))
        nn.init.kaiming_uniform_(self.weight1, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight2, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight3, nonlinearity='relu')

        # Bias terms
        self.bias1 = nn.Parameter(torch.Tensor(32))
        self.bias2 = nn.Parameter(torch.Tensor(64))
        self.bias3 = nn.Parameter(torch.Tensor(128))
        self.bias1.data.fill_(0)
        self.bias2.data.fill_(0)
        self.bias3.data.fill_(0)

        # Offset for deformable convolution
        self.offsets1 = nn.Conv2d(3, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)
        self.offsets2 = nn.Conv2d(32, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)
        self.offsets3 = nn.Conv2d(64, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dummy forward pass to determine the size for the fully connected layer
        dummy_input = torch.zeros(1, 3, 299, 299)  # Adjust the size based on your input image size
        fc_size = self._get_conv_output(dummy_input)

        # Fully connected layers
        self.fc1 = nn.Linear(fc_size, 512)
        self.fc2 = nn.Linear(512, 102)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def _get_conv_output(self, x):
        x = torchvision.ops.deform_conv2d(x, self.offsets1(x), self.weight1, self.bias1, padding=1)
        x = self.pool(x)
        x = torchvision.ops.deform_conv2d(x, self.offsets2(x), self.weight2, self.bias2, padding=1)
        x = self.pool(x)
        x = torchvision.ops.deform_conv2d(x, self.offsets3(x), self.weight3, self.bias3, padding=1)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        x = torchvision.ops.deform_conv2d(x, self.offsets1(x), self.weight1, self.bias1, padding=1)
        x = self.relu(x)
        x = self.pool(x)
        x = torchvision.ops.deform_conv2d(x, self.offsets2(x), self.weight2, self.bias2, padding=1)
        x = self.relu(x)
        x = self.pool(x)
        x = torchvision.ops.deform_conv2d(x, self.offsets3(x), self.weight3, self.bias3, padding=1)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
## Set device
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
model = DeformableCNNModel().to(device)


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
        torch.save(model.state_dict(), 'Models/deform.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Scheduler step (after validation)
    scheduler.step(valid_loss)  #

# Save the model if not done in the early stopping
if no_improvement_epochs < early_stopping_patience:
    torch.save(model.state_dict(), 'Models/deform.pth')

# Writing results to a CSV file
with open('Results/deform_results.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train Loss', 'Train Accuracy','Validation Loss', 'Validation Accuracy'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Training results saved to training_results.csv")