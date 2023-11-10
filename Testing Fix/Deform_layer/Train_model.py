import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import LabelEncoder
import torchvision
from tqdm import tqdm  # Standard import for scripts and applications
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
train_data = datasets.ImageFolder('../dataset/splitted_dataset/train', transform=train_transforms)
valid_data = datasets.ImageFolder('../dataset/splitted_dataset/valid', transform=valid_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
validloader = DataLoader(valid_data, batch_size=32)


# Get class names from the 'class_to_idx' attribute of the dataset
class_names = list(train_data.class_to_idx.keys())

# Initialize and fit the LabelEncoder with the class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Save the fitted LabelEncoder to a file for later use during inference
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

train_losses = []  # To store training losses
train_accuracies = []  # To store training accuracies
valid_losses = []  # To store validation losses
valid_accuracies = []  # To store validation accuracies

# Initialize Inception V3 with pre-trained weights
model = DeformableCNNModel().to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Initialize early stopping parameters
best_val_loss = np.Inf
patience = 5
patience_counter = 0
# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    total_train = 0
    correct_train = 0
    model.train()  # Set the model to training mode
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation loop...
    valid_loss = 0
    accuracy = 0
    total_valid = 0
    correct_valid = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Again, we access the main output for the loss
            valid_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    # Print out the losses and accuracy
    valid_loss = valid_loss / len(validloader.dataset)
    valid_losses.append(valid_loss / len(validloader))
    valid_accuracy = 100 * correct_valid / total_valid
    valid_accuracies.append(valid_accuracy)

    print(f"Validation Loss: {valid_loss / len(validloader):.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

    # Early Stopping
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_deform.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
            break

    # Scheduler step (after validation)
    scheduler.step()

if no_improvement_epochs < early_stopping_patience:
    torch.save(inception.state_dict(), 'deform.pth')

# After the training loop
df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Training Loss': train_losses,
    'Training Accuracy': train_accuracies,
    'Validation Loss': valid_losses,
    'Validation Accuracy': valid_accuracies
})

# Save the DataFrame to a CSV file
csv_file = 'Results/deform.csv'
df.to_csv(csv_file, index=False)

print(f'Training data saved to {csv_file}')