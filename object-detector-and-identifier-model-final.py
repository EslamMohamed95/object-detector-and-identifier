import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class CustomModel(nn.Module):
    def __init__(self, num_species=9):
        super(CustomModel, self).__init__()
        # Choose a backbone architecture (e.g., ResNet, MobileNet, DenseNet, or EfficientNet)
        self.backbone = models.resnet50(pretrained=True)

        # Define the head of the model to extract features and output predictions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

        # Head for object detection
        self.object_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 1),  # Output whether object present or not
            nn.Sigmoid()
        )
        # Head for cat or dog detection
        self.cat_or_dog_head = nn.Sequential(
            nn.Linear(num_features, 1),  # Output whether cat or dog
        )
        # Head for species classification
        self.species_head = nn.Sequential(
            nn.Linear(num_features, num_species),  # Output species probabilities
            nn.Softmax(dim=1)
        )

        # Bounding box head
        self.bbox_head = nn.Linear(num_features, 4)  # Output bounding box coordinates

    def forward(self, x):
        features = self.backbone(x)
        object_present = self.object_head(features)
        cat_or_dog = self.cat_or_dog_head(features)
        species_probs = self.species_head(features)
        bbox = self.bbox_head(features)

        return object_present, cat_or_dog, species_probs, bbox


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentation=None):

        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.image_folder = os.path.join(root_dir, 'images')
        self.label_folder = os.path.join(root_dir, 'labels')
        self.image_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.label_list = [f for f in os.listdir(self.label_folder) if f.endswith('.xml')]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.augmentation:
            image = self.augmentation(image)

        label_path = os.path.join(self.label_folder, self.image_list[idx].replace('.jpg', '.xml'))
        if os.path.exists(label_path):
            label = self._parse_xml(label_path)
        else:
            label = {
                'object_presence': torch.tensor([0.0]),
                'cat_or_dog': torch.tensor([0.0]),
                'bounding_box': torch.tensor([0.0, 0.0, 0.0, 0.0]),
                'species_probabilities': torch.zeros(9)
            }

        return image, label

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_presence = torch.tensor([1.0])
        cat_or_dog = torch.zeros(2)
        cat_or_dog[self._get_type_index(root.find('object/type').text)] = 1.0
        bounding_box = [float(root.find('object/bndbox/xmin').text),
                        float(root.find('object/bndbox/ymin').text),
                        float(root.find('object/bndbox/xmax').text),
                        float(root.find('object/bndbox/ymax').text)]
        species_probabilities = torch.zeros(9)
        species_probabilities[self._get_species_index("_".join(root.find('filename').text.split("_")[:-1]))] = 1.0

        return {
            'object_presence': object_presence,
            'cat_or_dog': cat_or_dog,
            'bounding_box': torch.tensor(bounding_box),
            'species_probabilities': species_probabilities
        }

    def _get_species_index(self, species):
        species_mapping = {
            'Abyssinian': 0,
            'Birman': 1,
            'Persian': 2,
            'american_bulldog': 3,
            'american_pit_bull_terrier': 4,
            'basset_hound': 5,
            'beagle': 6,
            'chihuahua': 7,
            'pomeranian': 8
        }
        return species_mapping[species]

    def _get_type_index(self, type):
        type_mapping = {
            'cat': 0,
            'dog': 1
        }
        return type_mapping[type]


class CustomLoss(nn.Module):
    def __init__(self, weight_object_present=1.0, weight_cat_or_dog=1.0, weight_species_probs=1.0, weight_bbox=1.0):
        super(CustomLoss, self).__init__()
        self.weight_object_present = weight_object_present
        self.weight_cat_or_dog = weight_cat_or_dog
        self.weight_species_probs = weight_species_probs
        self.weight_bbox = weight_bbox

        self.object_present_loss = nn.BCELoss()
        self.cat_or_dog_loss = nn.BCELoss()
        self.species_probs_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.MSELoss()

    def forward(self, outputs, labels):
        object_present_pred, cat_or_dog_pred, species_probs_pred, bbox_pred = outputs
        object_present_label = labels['object_presence']
        cat_or_dog_label = labels['cat_or_dog']
        species_probs_label = labels['species_probabilities']
        bbox_label = labels['bounding_box']

        # Calculate the loss for each component
        object_present_loss = self.object_present_loss(object_present_pred, object_present_label)
        cat_or_dog_loss = self.cat_or_dog_loss(cat_or_dog_pred, cat_or_dog_label)
        species_probs_loss = self.species_probs_loss(species_probs_pred, species_probs_label.argmax(dim=1))
        bbox_loss = self.bbox_loss(bbox_pred, bbox_label)

        # Weighted sum of the losses
        total_loss = (self.weight_object_present * object_present_loss +
                      self.weight_cat_or_dog * cat_or_dog_loss +
                      self.weight_species_probs * species_probs_loss +
                      self.weight_bbox * bbox_loss)

        return total_loss


def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    object_presence_preds = []
    object_presence_labels = []

    cat_or_dog_preds = []
    cat_or_dog_labels = []

    species_preds = []
    species_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}

        with torch.no_grad():
            object_present_pred, cat_or_dog_pred, species_probs_pred, bbox_pred = model(images)

        object_presence_pred = (object_present_pred >= 0.5).float()
        object_presence_preds.extend(object_presence_pred.cpu().numpy())
        object_presence_labels.extend(labels['object_presence'].cpu().numpy())

        cat_or_dog_pred = (cat_or_dog_pred >= 0.5).float()
        cat_or_dog_preds.extend(cat_or_dog_pred.cpu().numpy())
        cat_or_dog_labels.extend(labels['cat_or_dog'].cpu().numpy())

        species_pred = species_probs_pred.argmax(dim=1).cpu().numpy()
        species_preds.extend(species_pred)
        species_labels.extend(labels['species_probabilities'].argmax(dim=1).cpu().numpy())

    # Calculate metrics for object presence
    object_presence_accuracy = accuracy_score(object_presence_labels, object_presence_preds)
    object_presence_precision = precision_score(object_presence_labels, object_presence_preds)
    object_presence_recall = recall_score(object_presence_labels, object_presence_preds)
    object_presence_f1 = f1_score(object_presence_labels, object_presence_preds)

    # Calculate metrics for cat or dog detection
    cat_or_dog_accuracy = accuracy_score(cat_or_dog_labels, cat_or_dog_preds)
    cat_or_dog_precision = precision_score(cat_or_dog_labels, cat_or_dog_preds)
    cat_or_dog_recall = recall_score(cat_or_dog_labels, cat_or_dog_preds)
    cat_or_dog_f1 = f1_score(cat_or_dog_labels, cat_or_dog_preds)

    # Calculate metrics for species classification
    species_accuracy = accuracy_score(species_labels, species_preds)
    species_precision = precision_score(species_labels, species_preds, average='weighted')
    species_recall = recall_score(species_labels, species_preds, average='weighted')
    species_f1 = f1_score(species_labels, species_preds, average='weighted')

    metrics = {
        'Object Presence Accuracy': object_presence_accuracy,
        'Object Presence Precision': object_presence_precision,
        'Object Presence Recall': object_presence_recall,
        'Object Presence F1': object_presence_f1,
        'Cat or Dog Accuracy': cat_or_dog_accuracy,
        'Cat or Dog Precision': cat_or_dog_precision,
        'Cat or Dog Recall': cat_or_dog_recall,
        'Cat or Dog F1': cat_or_dog_f1,
        'Species Accuracy': species_accuracy,
        'Species Precision': species_precision,
        'Species Recall': species_recall,
        'Species F1': species_f1
    }

    return metrics


def train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs, resume_from=None):
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    criterion = CustomLoss()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Calculate average training loss
        train_loss /= len(train_dataloader.dataset)

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_dataloader)

        # Save best weights
        val_loss = val_metrics['Loss']  # Assuming 'Loss' is one of the metrics in evaluate_model()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, 'best_weights.pth')

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss, val_loss))


def visualize(model, image_folder, weight_file, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = model()  # Replace with your actual model class
    model.load_state_dict(torch.load(weight_file))
    model.to(device)
    model.eval()

    image_details = {}

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        if not os.path.isfile(image_path) or not image_name.lower().endswith(('.jpg', '.jpeg')):
            continue

        # Load and preprocess the image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(image_tensor)

        # Process the model outputs
        object_presence_pred, cat_or_dog_pred, species_probs_pred, bbox_pred = outputs

        object_presence = object_presence_pred.item() > 0.5
        cat_or_dog = "cat" if cat_or_dog_pred.item() >= 0.5 else "dog"

        if object_presence:
            species = species_probs_pred.argmax(dim=1).item()
        else:
            species = -1

        bbox = bbox_pred.squeeze().tolist() if object_presence else [0, 0, 0, 0]

        image_detail = {
            "has_object": object_presence,
            "cat_or_dog": cat_or_dog if object_presence else "NA",
            "species": species if object_presence else "NA",
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3]
        }
        image_details[image_name] = image_detail

        # Draw bounding box on image and save it
        draw = ImageDraw.Draw(image)
        if object_presence:
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2)
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)

    return image_details


def main():
    # Set the required variables
    image_folder = './dataset/images'
    output_folder = './output/folder'
    weight_file = './model/weights.pth'
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    resume_from = None

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create a list of image paths
    image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder) if
                   img_name.lower().endswith(('.jpg', '.jpeg'))]

    # Split the images for training and validation
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    # Create the datasets
    train_dataset = ImageFolder(image_folder, transform=transform, loader=torchvision.datasets.folder.default_loader)
    train_dataset.samples = train_paths

    val_dataset = ImageFolder(image_folder, transform=transform, loader=torchvision.datasets.folder.default_loader)
    val_dataset.samples = val_paths

    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CustomModel()
    model.to(device)

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs, resume_from)

    # Visualize the images and save outputs
    image_details = visualize(model, image_folder, weight_file, output_folder)
    print(image_details)


if __name__ == '__main__':
    main()
