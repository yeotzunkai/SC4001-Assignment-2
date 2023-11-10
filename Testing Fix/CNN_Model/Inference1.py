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
from sklearn.preprocessing import LabelEncoder
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_dir = 'splitted_dataset/test'

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


# Load the model
model = CNNModel()
model = torch.load('model.pth', map_location=device)
model.to(device)  # Move the model to the specified device
model.eval()

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    with open('cat_to_name.json') as f:
        idx_to_class = json.load(f)

    adjusted_predicted = predicted.item() + 1
    return idx_to_class[str(adjusted_predicted)]

def get_random_images_from_folder(folder_path, num_images=10):
    categories = os.listdir(folder_path)
    selected_images = []
    actual_labels = []

    for _ in range(num_images):
        category = random.choice(categories)
        category_path = os.path.join(folder_path, category)
        image_file = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, image_file)
        selected_images.append(image_path)
        actual_labels.append(category)

    return selected_images, actual_labels

def get_all_images_from_folder(folder_path):
    category_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    image_paths = []
    actual_labels = []

    for category_folder in category_folders:
        category = os.path.basename(category_folder)
        for image_file in os.listdir(category_folder):
            image_path = os.path.join(category_folder, image_file)
            image_paths.append(image_path)
            actual_labels.append(category)
    
    return image_paths, actual_labels


# Get random test images 
#test_images, actual_labels = get_random_images_from_folder(test_data_dir, num_images=10)
# Get all images
test_images, actual_labels = get_all_images_from_folder(test_data_dir)

predicted_labels = []
for image_path in test_images:
    predicted_class = predict_image(image_path)
    predicted_labels.append(predicted_class)

# Predict labels for all test images
predicted_labels = [predict_image(image_path) for image_path in test_images]

actual_labels_encoded = label_encoder.transform(actual_labels)

# Calculate accuracy
accuracy = accuracy_score(actual_labels_encoded, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
'''
# Number of images to display
num_images_to_display = len(test_images)

# Define the number of columns (images per row)
num_columns = 5
num_rows = (num_images_to_display + num_columns - 1) // num_columns

# Set up the matplotlib figure and axes
fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
# Optional: Displaying images with predicted and actual labels

for i in range(num_columns * num_rows):
    axes[i].axis('off')  # Initially turn off all axes

# Plot each image with its predictions
for i, (image_path, predicted_label, actual_label) in enumerate(zip(test_images, predicted_labels, actual_labels)):
    ax = axes[i]
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
    ax.axis('on')  # Only turn on the axes that are used

plt.tight_layout()
plt.show()
'''