import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import LabelEncoder
import csv
from torch.optim.lr_scheduler import CyclicLR
import math
import matplotlib.pyplot as plt

class CLR(object):
    def __init__(self, optimizer, bn, base_lr=1e-7, max_lr=100):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.optim = optimizer
        self.bn = bn - 1
        ratio = self.max_lr / self.base_lr
        self.mult = ratio ** (1 / self.bn)
        self.best_loss = 1e9
        self.iteration = 0
        self.lrs = []
        self.losses = []

    def calc_lr(self, loss):
        self.iteration += 1
        if math.isnan(loss) or loss > 4 * self.best_loss:
            return -1
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss

        mult = self.mult ** self.iteration
        lr = self.base_lr * mult

        self.lrs.append(lr)
        self.losses.append(loss)

        return lr

    def plot(self, start=10, end=-5):
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log')

    def plot_lr(self):
        plt.xlabel("Iterations")
        plt.ylabel("Learning Rate")
        plt.plot(self.lrs)
        plt.yscale('log')


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(299),  # Inception V3 expects 299x299 input
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(320),  # Slightly larger than 299 for cropping
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder('splitted_dataset/train', transform=train_transforms)
valid_data = datasets.ImageFolder('splitted_dataset/test', transform=valid_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
validloader = DataLoader(valid_data, batch_size=64)


# Get class names from the 'class_to_idx' attribute of the dataset
class_names = list(train_data.class_to_idx.keys())

# Initialize and fit the LabelEncoder with the class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Save the fitted LabelEncoder to a file for later use during inference
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)


# Initialize Inception V3 with pre-trained weights
inception = models.inception_v3(pretrained=True, aux_logits=True)

# Freeze model parameters
for param in inception.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
# Inception V3's aux_logits=True by default which has an auxiliary output. Generally, it should be turned off by setting to False
# when adapting the model to a new task
num_ftrs = inception.fc.in_features
inception.fc = nn.Linear(num_ftrs, 102)  # Assuming 102 flower classes

# Move the model to the GPU if available
inception = inception.to(device)
'''
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(inception.fc.parameters(), lr=0.001)
optimizer = optim.Adam(inception.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
'''
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(inception.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(inception.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Early stopping parameters
early_stopping_patience = 15
min_val_loss = float('inf')
no_improvement_epochs = 0

# List to store epoch results
results = []

# Training loop
epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    inception.train()  # Set the model to training mode
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = inception(inputs)
        # If aux_logits is True, the output will be an InceptionOutputs object
        # We need to access the main output for computing the loss
        main_outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(main_outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

         # Calculate training accuracy
        _, predicted = torch.max(main_outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
       

    train_accuracy = correct_train / total_train

    # Validation loop...
    valid_loss = 0
    correct_val = 0
    total_val = 0
    inception.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inception(inputs)
            # Again, we access the main output for the loss
            main_outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            valid_loss += criterion(main_outputs, labels).item()
            
            _, preds = torch.max(main_outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    valid_accuracy = correct_val / total_val
     # Print out the losses and accuracy
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Train accuracy: {train_accuracy:.3f}.. "
          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {valid_accuracy:.3f}")

    # Append the results of this epoch to the results list
    results.append({
        'Epoch': epoch + 1,
        'Train Loss': running_loss / len(trainloader),
        'Train Accuracy': train_accuracy,
        'Validation Loss': valid_loss / len(validloader),
        'Validation Accuracy': valid_accuracy
    })

    # Early stopping check
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        no_improvement_epochs = 0
        # Save the model only when validation loss decreases
        torch.save(inception.state_dict(), 'inception_flower_model2.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Scheduler step (after validation)
    scheduler.step()


# Save the model if not done in the early stopping
if no_improvement_epochs < early_stopping_patience:
    torch.save(inception.state_dict(), 'inception_flower_model2.pth')

# Writing results to a CSV file
with open('training_results2.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train Loss', 'Train Accuracy','Validation Loss', 'Validation Accuracy'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Training results saved to training_results2.csv")