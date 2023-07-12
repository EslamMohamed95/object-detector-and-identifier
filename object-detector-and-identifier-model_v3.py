import os
import xml.etree.ElementTree as ET

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomModel(nn.Module):
    def __init__(self, num_species=9):
        super(CustomModel, self).__init__()
        # Choose a backbone architecture (e.g., ResNet, MobileNet, or EfficientNet)
        self.backbone = models.resnet50(pretrained=True)

        # Define the head of the model to extract features and output predictions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

        # Additional layers for object presence and bounding box regression
        self.object_presence = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.bounding_box_regression = nn.Linear(num_features, 4)

        # Additional layers for species classification
        self.species_classification = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_species),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Forward pass through the model
        out_backbone = self.backbone(x)
        object_presence = self.object_presence(out_backbone)
        bounding_box = self.bounding_box_regression(out_backbone)
        species_probabilities = self.species_classification(out_backbone)

        return object_presence, bounding_box, species_probabilities


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

        label_path = os.path.join(self.label_folder, self.image_list[idx].replace('.jpg', '.xml'))
        if os.path.exists(label_path):
            label = self._parse_xml(label_path)
        else:
            label = {
                'object_presence': torch.tensor([0.0]),
                'bounding_box': torch.tensor([0.0, 0.0, 0.0, 0.0]),
                'species_probabilities': torch.zeros(9)
            }

        return image, label

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_presence = torch.tensor([1.0])
        # cat_or_dog = torch.zeros(2)
        # cat_or_dog[root.find('object/name').text] = 1.0
        bounding_box = [float(root.find('object/bndbox/xmin').text),
                        float(root.find('object/bndbox/ymin').text),
                        float(root.find('object/bndbox/xmax').text),
                        float(root.find('object/bndbox/ymax').text)]
        species_probabilities = torch.zeros(9)
        species_probabilities[self._get_species_index("_".join(root.find('filename').text.split("_")[:-1]))] = 1.0

        return {
            'object_presence': object_presence,
            # 'cat_or_dog': cat_or_dog,
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


def visualize_output(image, object_presence, bounding_box, species_probabilities):
    # Convert tensor to numpy array and reshape if necessary
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Display bounding box if object is present
    if object_presence > 0.5:
        xmin, ymin, xmax, ymax = bounding_box.tolist()
        width = xmax - xmin
        height = ymax - ymin

        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)

    # Add species labels
    species_labels = ['Abyssinian', 'Birman', 'Persian', 'american_bulldog',
                      'american_pit_bull_terrier', 'basset_hound', 'beagle',
                      'chihuahua', 'pomeranian']
    species_probabilities = species_probabilities.tolist()
    species_indices = [i for i, prob in enumerate(species_probabilities) if prob > 0.5]
    species_names = [species_labels[i] for i in species_indices]

    # Add species labels to the plot
    for i, species_name in enumerate(species_names):
        ax.text(10, 10 + i * 20, species_name, color='white', fontsize=8,
                bbox=dict(facecolor='black', edgecolor='none'))

    # Show the plot
    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            object_presence, bounding_box, species_probabilities = model(images)

            # Compute loss
            object_presence_loss = criterion(object_presence, labels['object_presence'].to(device))
            bounding_box_loss = bbox_criterion(bounding_box, labels['bounding_box'].to(device))
            species_loss = species_criterion(species_probabilities,
                                             torch.argmax(labels['species_probabilities'], dim=1).to(device))
            loss = object_presence_loss + bounding_box_loss + species_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                object_presence, bounding_box, species_probabilities = model(images)

                # Compute loss
                object_presence_loss = criterion(object_presence, labels['object_presence'].to(device))
                bounding_box_loss = bbox_criterion(bounding_box, labels['bounding_box'].to(device))
                species_loss = species_criterion(species_probabilities,
                                                 torch.argmax(labels['species_probabilities'], dim=1).to(device))
                loss = object_presence_loss + bounding_box_loss + species_loss

                val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model
def test_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            object_presence, bounding_box, species_probabilities = model(images)

            # Compute loss
            loss = criterion(object_presence, labels['object_presence'].to(device)) + \
                   bbox_criterion(bounding_box, labels['bounding_box'].to(device)) + \
                   species_criterion(species_probabilities,
                                     torch.argmax(labels['species_probabilities'], dim=1).to(device))

            test_loss += loss.item() * images.size(0)

            # Count correct predictions
            predicted_presence = (object_presence > 0.5).int()
            correct_predictions += (predicted_presence == labels['object_presence']).sum().item()

            total_samples += images.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct_predictions / total_samples

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset
dataset = CustomDataset(root_dir='./dataset', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = CustomModel(num_species=9).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss for object presence
bbox_criterion = nn.MSELoss()  # Mean squared error loss for bounding box regression
species_criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for species classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Call the train_model function
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(trained_model.state_dict(), 'trained_model.pth')


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the path to your image
image_path = "/content/chihuahua-g960eb0099_640.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((245, 280)),
    transforms.ToTensor(),
])
input_image = transform(image).unsqueeze(0)

# Load the trained model
model.load_state_dict(torch.load("/content/trained_model.pth"))
model.to(torch.device("cuda"))  # Move the model to the GPU device
model.eval()

# Move the input image tensor to the same device as the model
input_image = input_image.to(torch.device("cuda"))

# Make predictions
with torch.no_grad():
    object_presence, bounding_box, species_probabilities = model(input_image)

# Convert the tensor outputs to numpy arrays
bounding_box = bounding_box.squeeze().cpu().numpy()
species_probabilities = species_probabilities.squeeze().cpu().numpy()

# Define the class labels for species
species_labels = [
    "Abyssinian", "Birman", "Persian",
    "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "chihuahua", "pomeranian"
]

# Visualize the image with bounding box and predicted species
plt.imshow(image)
plt.axis("off")

# Draw bounding box
xmin, ymin, xmax, ymax = bounding_box
plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, color="red", linewidth=2))

# Get the predicted species
predicted_species = species_labels[species_probabilities.argmax()]

plt.text(xmin, ymin, predicted_species, color="red",
         fontsize=8, bbox=dict(facecolor="white", alpha=0.8))

# Show the image with bounding box and predicted species
plt.show()
