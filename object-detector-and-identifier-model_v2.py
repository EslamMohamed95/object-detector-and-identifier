# Required Libraries
import os

# Visualize Function
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from torch import utils
from torch.utils.data import DataLoader
from torchvision import models, transforms


# Model Architecture
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        self.classifier = nn.Linear(2048, num_classes)
        self.bbox_regressor = nn.Linear(2048, 4)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        classes = self.classifier(features)
        bboxes = self.bbox_regressor(features)
        return classes, bboxes


def read_xml_file(path):
    with open(path, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    return {
        "cat_or_dog": bs_data.find("name").text,
        "xmin": int(bs_data.find("xmin").text),
        "ymin": int(bs_data.find("ymin").text),
        "xmax": int(bs_data.find("xmax").text),
        "ymax": int(bs_data.find("ymax").text),
        "specie": "_".join(path.split(os.sep)[-1].split("_")[:-1])
    }


# Custom Dataloader

class CustomDataset:
    def __init__(self, dataset_path, images_list, train=False):

        self.image_dim = (224, 224)
        self.species_map = {'NA': 0, 'pomeranian': 1, 'Persian': 2, 'beagle': 3, 'Abyssinian': 4,
                            'american_pit_bull_terrier': 5, 'basset_hound': 6, 'american_bulldog': 7, 'Birman': 8,
                            'chihuahua': 9}
        self.preprocess = []

        image_folder_path = os.path.join(dataset_path, "images")
        label_folder_path = os.path.join(dataset_path, "labels")

        for path in os.listdir(image_folder_path):
            name = path.split(os.sep)[-1].split(".")[0]

            if name in images_list:
                img_fname = [i for i in os.listdir('dataset/images/') if name in i][0]
                img_path = os.path.join(image_folder_path, img_fname)

                try:
                    xml_path = os.path.join(label_folder_path, name + ".xml")
                    xml_data = read_xml_file(xml_path)

                except FileNotFoundError:
                    xml_data = {'cat_or_dog': 'NA', 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0, 'specie': 'NA'}

                xml_data['specie'] = self.species_map[xml_data['specie']]  # map species class to an integer identifier
                self.preprocess.append([img_path, xml_data])

    def __len__(self):
        return len(self.preprocess)

    def __getitem__(self, index):
        image_path, labels = self.preprocess[index]
        labels = [float(labels['xmin']), float(labels['ymin']), float(labels['xmax']), float(labels['ymax']),
                  float(labels['specie'])]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_dim)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        labels = torch.tensor([labels])

        if torch.cuda.is_available():
            return image.type(torch.cuda.FloatTensor), torch.squeeze(labels)
        else:
            return image, torch.squeeze(labels)


# Data Augmentation and Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Loss Function
class ObjectDetectionLoss(nn.Module):
    def __init__(self):
        super(ObjectDetectionLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.MSELoss()

    def forward(self, classes, bboxes, targets):
        class_loss = self.class_loss(classes, targets)
        bbox_loss = self.bbox_loss(bboxes, targets)
        return class_loss + bbox_loss


# Test Function
def test(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_objects = 0
    correct_objects = 0
    total_cats = 0
    correct_cats = 0
    total_dogs = 0
    correct_dogs = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            classes, bboxes = model(images)

            # Compute metrics
            total_objects += images.size(0)
            _, predicted_classes = torch.max(classes.data, 1)
            correct_objects += (predicted_classes == targets).sum().item()

            # Calculate metrics for cats and dogs
            cat_indices = targets < 3
            dog_indices = targets >= 3
            total_cats += cat_indices.sum().item()
            total_dogs += dog_indices.sum().item()

            correct_cats += (predicted_classes[cat_indices] == targets[cat_indices]).sum().item()
            correct_dogs += (predicted_classes[dog_indices] == targets[dog_indices]).sum().item()

    accuracy = correct_objects / total_objects
    cat_accuracy = correct_cats / total_cats
    dog_accuracy = correct_dogs / total_dogs

    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Cat accuracy: {cat_accuracy:.2%}")
    print(f"Dog accuracy: {dog_accuracy:.2%}")


# Training
def train(model, dataloader, loss_function, optimizer, device, resume=False, num_epochs=10):
    if resume:
        checkpoint = torch.load('best_weights.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
    else:
        start_epoch = 0
        best_loss = float('inf')

    model.to(device)
    model.train()
    for epoch in range(start_epoch, num_epochs):
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            classes, bboxes = model(images)
            loss = loss_function(classes, bboxes, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint if current loss is the best
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, 'best_weights.pth')


def visualize(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    preprocessed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Perform inference
        classes, bboxes = model(preprocessed_image)

        # Extract predicted class and bounding box
        _, predicted_class = torch.max(classes.data, 1)
        predicted_bbox = bboxes[0].cpu().numpy()

    # Visualize the results
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_with_bbox = cv2.rectangle(
        img,
        (int(predicted_bbox[0]), int(predicted_bbox[1])),
        (int(predicted_bbox[2]), int(predicted_bbox[3])),
        (0, 255, 0),
        2
    )
    plt.imshow(img_with_bbox)
    plt.title(f"Predicted Class: {predicted_class.item()}")
    plt.axis('off')
    plt.show()


# Main Function
def main():
    # Hyperparameters
    num_classes = 9
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # CUSTOM DATALOADER IMPLEMENTATION
    train_list = np.load('./dataset/train_list.npy', allow_pickle=True).tolist()
    val_list = np.load('./dataset/val_list.npy', allow_pickle=True).tolist()

    # Data loading
    training_dataset = CustomDataset("dataset/", images_list=train_list)
    train_dataloaders = utils.data.DataLoader(training_dataset, batch_size=16, shuffle=True)

    val_dataset = CustomDataset("dataset/", images_list=val_list)
    val_dataloaders = utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    # Model initialization
    model = ObjectDetectionModel(num_classes)

    # Loss function initialization
    loss_function = ObjectDetectionLoss()

    # Optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    train(model, train_dataloaders, loss_function, optimizer, device)

    # Testing
    test(model, val_dataloaders)

    # Visualization
    image = ...  # Load an image
    visualize(model, image)


# Run the main function
if __name__ == "__main__":
    main()
