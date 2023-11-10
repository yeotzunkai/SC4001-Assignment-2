import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import LabelEncoder



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
valid_data = datasets.ImageFolder('splitted_dataset/valid', transform=valid_transforms)

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

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(inception.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Early stopping parameters
early_stopping_patience = 5
min_val_loss = float('inf')
no_improvement_epochs = 0

# Training loop
epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
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

    # Validation loop...
    valid_loss = 0
    accuracy = 0
    inception.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inception(inputs)
            # Again, we access the main output for the loss
            main_outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            valid_loss += criterion(main_outputs, labels).item()
            
            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels.data).item()
    
    # Print out the losses and accuracy
    valid_loss = valid_loss / len(validloader.dataset)
    accuracy = accuracy / len(validloader.dataset)
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation loss: {valid_loss:.3f}.. "
          f"Validation accuracy: {accuracy:.3f}")

    # Early stopping check
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        no_improvement_epochs = 0
        # Save the model only when validation loss decreases
        torch.save(inception.state_dict(), 'inception_flower_model.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Scheduler step (after validation)
    scheduler.step()

# Save the model if not done in the early stopping
if no_improvement_epochs < early_stopping_patience:
    torch.save(inception.state_dict(), 'inception_flower_model.pth')