import torch
from torchvision import models, transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_dir = '../dataset/splitted_dataset/test'

# Initialize the Inception V3 model with pre-trained weights
inception = models.inception_v3(pretrained=True, aux_logits=True)
num_ftrs = inception.fc.in_features
inception.fc = torch.nn.Linear(num_ftrs, 102)  # Replace the final fully connected layer

inception.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),  # Increased dropout
    torch.nn.Linear(512, 102),
)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

# Apply weight initialization
inception.fc.apply(initialize_weights)

# Load the saved state dictionary into the model
state_dict = torch.load('../Models/inception_flower_model.pth', map_location=device)
inception.load_state_dict(state_dict)

# Set the model to evaluation mode
inception = inception.to(device)
inception.eval()

# Load the label encoder
with open('../label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Define the image transformations
inference_transforms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

### RUN BELOW FOR ACCURACY SCORE ON ALL IMAGES

# Function to perform inference
def predict_image(image_path, model):
    image = Image.open(image_path)
    image = inference_transforms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.cpu().numpy()[0]  # Move to CPU and convert to numpy

    return predicted_class


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

# Get all images and their actual labels
test_images, actual_labels = get_all_images_from_folder(test_data_dir)

# Predict labels for all test images
predicted_labels = [predict_image(image_path, inception) for image_path in test_images]

# Encode actual folder names to the same label space as predicted
# Ensure that the classes are encoded in the same way they were during training
actual_labels_encoded = label_encoder.transform(actual_labels)

# Calculate accuracy
accuracy = accuracy_score(actual_labels_encoded, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")


## RUN BELOW FOR IMAGE RESULTS ON FEW RANDOM IMAGES

# Function to perform inference
def predict_image(image_path, model):
    image = Image.open(image_path)
    image = inference_transforms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.cpu().numpy()[0]  # Move to CPU and convert to numpy

    return predicted_class

# Function to display an image and its label
def display_image(image_path, actual_label, predicted_label):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
    plt.axis('off')

# Get random images from the test folder
def get_random_test_images(test_folder, num_images=5):
    all_classes = os.listdir(test_folder)
    selected_images = []
    actual_labels = []
    for _ in range(num_images):
        actual_class = random.choice(all_classes)
        class_folder = os.path.join(test_folder, actual_class)
        image_name = random.choice(os.listdir(class_folder))
        image_path = os.path.join(class_folder, image_name)
        selected_images.append(image_path)
        actual_labels.append(actual_class)
    return selected_images, actual_labels

# Path to the test folder
test_folder = '../dataset/splitted_dataset/test'
random_images, actual_labels = get_random_test_images(test_folder, num_images=5)

# Plot the images with predictions
plt.figure(figsize=(15, 5))
for idx, (image_path, actual_label) in enumerate(zip(random_images, actual_labels)):
    predicted_index = predict_image(image_path, inception)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]  # Get the class name from the encoder
    plt.subplot(1, 5, idx+1)
    display_image(image_path, actual_label, predicted_label)

plt.tight_layout()
plt.show()
