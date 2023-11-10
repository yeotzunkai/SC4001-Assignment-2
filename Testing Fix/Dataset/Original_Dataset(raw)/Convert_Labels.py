import json
import scipy.io
import os
import shutil
from sklearn.model_selection import train_test_split


def load_labels(label_path):
    mat = scipy.io.loadmat(label_path)
    return mat['labels'][0] - 1  # Adjusting labels to start from 0

def load_category_names(json_path):
    with open(json_path, 'r') as f:
        category_names = json.load(f)
    return category_names

def split_dataset(image_folder, labels, test_size=0.2, valid_size=0.1):
    image_paths = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.jpg')]
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42)
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=valid_size, random_state=42)
    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels

def create_and_save_datasets(image_paths, labels, base_folder, category_names):
    for split in ['train', 'valid', 'test']:
        split_folder = os.path.join(base_folder, split)
        os.makedirs(split_folder, exist_ok=True)
        
        split_paths = image_paths[split]
        split_labels = labels[split]

        for path, label in zip(split_paths, split_labels):
            category_folder = os.path.join(split_folder, category_names[str(label + 1)])  # Adjusting index for 1-based label
            os.makedirs(category_folder, exist_ok=True)
            shutil.copy(path, category_folder)

# Set your file paths and folder here
image_folder = 'jpg'
label_path = 'imagelabels.mat'
json_path = 'cat_to_name.json'
output_folder = 'splitted_dataset'

labels = load_labels(label_path)
category_names = load_category_names(json_path)

# Splitting dataset
train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels = split_dataset(image_folder, labels)

# Preparing data structure for copying files
image_paths = {'train': train_paths, 'valid': valid_paths, 'test': test_paths}
labels = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}

# Creating and saving datasets
create_and_save_datasets(image_paths, labels, output_folder, category_names)