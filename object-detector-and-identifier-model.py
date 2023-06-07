
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, num_species=9):
        super(Model, self).__init__()

        pretrained_model = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])

        self.num_species = num_species

        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.have_object = nn.Linear(512, 1)
        self.cat_or_dog = nn.Linear(512, 1)
        self.specie = nn.Linear(512, num_species)
        self.bbox = nn.Linear(512, 4)

    def forward(self, input):
        out_backbone = self.backbone(input)
        out_backbone = out_backbone.view(out_backbone.size(0), -1)

        out = F.relu(self.fc(out_backbone))

        have_object = torch.sigmoid(self.have_object(out))
        cat_or_dog = torch.sigmoid(self.cat_or_dog(out))
        specie = nn.functional.softmax(self.specie(out), dim=1)
        bbox = torch.sigmoid(self.bbox(out))

        bbox = bbox.view(-1, 4)
        bbox[:, 0] *= input.size(3)  # xmin
        bbox[:, 1] *= input.size(2)  # ymin
        bbox[:, 2] *= input.size(3)  # xmax
        bbox[:, 3] *= input.size(2)  # ymax

        return {
            "bbox": bbox,
            "object": have_object,
            "cat_or_dog": cat_or_dog,
            "specie": specie
        }


def read_xml_file(path):
    try:
        with open(path, 'r') as f:
            data = f.read()
        bs_data = BeautifulSoup(data, 'xml')
        return {
            "has_object": True,
            "cat_or_dog": bs_data.find("name").text,
            "xmin": int(bs_data.find("xmin").text),
            "ymin": int(bs_data.find("ymin").text),
            "xmax": int(bs_data.find("xmax").text),
            "ymax": int(bs_data.find("ymax").text),
            "specie": "_".join(path.split(os.sep)[-1].split("_")[:-1]),
            "bbox": [int(bs_data.find("xmin").text), int(bs_data.find("ymin").text), int(bs_data.find("xmax").text),
                     int(bs_data.find("ymax").text)]
        }
    except FileNotFoundError:
        return {
            "has_object": False,
            "cat_or_dog": "NA",
            "xmin": 0,
            "ymin": 0,
            "xmax": 0,
            "ymax": 0,
            "specie": "NA",
            "bbox": [0, 0, 0, 0]

        }


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, images_list, train=False):
        self.dataset_path = dataset_path
        self.images_list = images_list
        self.train = train

        self.image_folder_path = os.path.join(self.dataset_path, "images")
        self.label_folder_path = os.path.join(self.dataset_path, "labels")

        self.transforms = transforms.Compose([
            transforms.Resize((200, 200)),  # Resize the image to 200x200
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        if self.train:
            self.augmentations = transforms.Compose([
                transforms.Resize((200, 200)),  # Resize the image to 200x200
                transforms.RandomCrop((150, 200)),  # Randomly crop the image to 150x200
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with probability 0.5
                transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        name = self.images_list[index]
        image_path = os.path.join(self.image_folder_path, name + ".jpg")
        label_path = os.path.join(self.label_folder_path, name + ".xml")

        image = Image.open(image_path).convert("RGB")
        label = read_xml_file(label_path)

        if self.train:
            image = self.augmentations(image)

        image = self.transforms(image)

        return image, label


def train_model(dataset_path, train_list, val_list, num_species, epochs, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_species)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    has_object_loss = nn.BCELoss()
    cat_or_dog_loss = nn.BCELoss()
    specie_loss = nn.CrossEntropyLoss()
    # bbox_loss = nn.MSELoss()
    xmin_loss = torch.nn.MSELoss()
    ymin_loss = torch.nn.MSELoss()
    xmax_loss = torch.nn.MSELoss()
    ymax_loss = torch.nn.MSELoss()

    train_dataset = CustomDataset(dataset_path, train_list, train=True)
    val_dataset = CustomDataset(dataset_path, val_list)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    writer = SummaryWriter()

    num_epochs = epochs

    best_val_loss = float("inf")  # Variable to track the best validation loss
    best_model_weights = None  # Variable to store the weights of the best model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels['cat_or_dog'] = torch.tensor([1 if x == 'dog' else 0 for x in labels['cat_or_dog']]).to(device)
            species = list(set(labels['specie']))
            labels['specie'] = torch.tensor([species.index(x) for x in labels['specie']]).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss_have_object = has_object_loss(outputs['object'], labels['has_object'].float().unsqueeze(1))
            loss_specie = specie_loss(outputs["specie"], labels['specie'])
            loss_cat_or_dog = cat_or_dog_loss(outputs["cat_or_dog"], labels['cat_or_dog'].float().unsqueeze(1))
            loss_xmin = xmin_loss(outputs["bbox"][:, 0], labels['xmin'])
            loss_ymin = ymin_loss(outputs["bbox"][:, 1], labels['ymin'])
            loss_xmax = xmax_loss(outputs["bbox"][:, 2], labels['xmax'])
            loss_ymax = ymax_loss(outputs["bbox"][:,3], labels['ymax'])

            loss = loss_have_object + loss_specie + loss_cat_or_dog + loss_xmin + loss_xmax + loss_ymin + loss_ymax

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        bbox_metric = torchmetrics.AverageMeter()
        object_metric = torchmetrics.AverageMeter()
        cat_or_dog_metric = torchmetrics.AverageMeter()
        specie_metric = torchmetrics.AverageMeter()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                species = list(set(labels['specie']))
                labels['specie'] = torch.tensor([species.index(x) for x in labels['specie']]).to(device)
                labels['cat_or_dog'] = torch.tensor([1 if x == 'dog' else 0 for x in labels['cat_or_dog']]).to(device)
                outputs = model(images)

                loss = 0.0
                for i, (key, value) in enumerate(outputs.items()):
                    loss += loss_functions[i](value, labels[key])

                val_loss += loss

                bbox_metric.update(outputs["bbox"], labels["bbox"])
                object_metric.update(outputs["object"], labels["object"])
                cat_or_dog_metric.update(outputs["cat_or_dog"], labels["cat_or_dog"])
                specie_metric.update(outputs["specie"], labels["specie"])

        val_loss /= len(val_loader.dataset)

        bbox_mse = torch.sqrt(torchmetrics.functional.mean_squared_error(
            bbox_metric.compute(), val_loader.dataset.bbox_transform.inverse_transform(bbox_metric.compute())
        ))

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("BBox MSE/validation", bbox_mse, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"BBox MSE: {bbox_mse:.4f}"
        )

        # Check if current validation loss is better than the previous best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    # Save the best model weights
    torch.save(best_model_weights, model_weights_path)

    writer.close()


def visualize(model_weights, image_folder_path, image_name, output_folder="output"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    image_path = None
    for extension in [".jpeg", ".jpg", ".png"]:
        image_path = os.path.join(image_folder_path, image_name + extension)
        if os.path.exists(image_path):
            break

    if image_path is None:
        print(f"Image {image_name} not found.")
        return

    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_image)

    # Extract the required output values
    bbox = output["bbox"].squeeze().tolist()
    object_pred = output["object"].item()
    cat_or_dog_pred = output["cat_or_dog"].item()
    specie_pred = output["specie"].item()

    # Prepare the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Draw bounding box on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=2)

    # Save the annotated image
    annotated_image_path = os.path.join(output_folder, image_name + "_annotated.jpg")
    image.save(annotated_image_path)

    return {
        "bbox": bbox,
        "object": object_pred,
        "cat_or_dog": cat_or_dog_pred,
        "specie": specie_pred,
        "annotated_image_path": annotated_image_path
    }


def validate_and_create_lists(dataset_path):
    images_folder_path = os.path.join(dataset_path, "images")
    labels_folder_path = os.path.join(dataset_path, "labels")

    image_files = sorted([f.split(".")[0] for f in os.listdir(images_folder_path) if f.endswith(".jpg")])
    label_files = sorted([f.split(".")[0] for f in os.listdir(labels_folder_path) if f.endswith(".xml")])

    assert image_files == label_files, "Mismatch between image and label files"

    random.seed(42)
    random.shuffle(image_files)

    num_images = len(image_files)
    num_train = int(0.8 * num_images)

    train_list = image_files[:num_train]
    val_list = image_files[num_train:]

    return train_list, val_list


if __name__ == "__main__":

    epochs = 100
    num_species = 9

    dataset_path = "./dataset"
    model_weights_path = "./model_weights.h5"
    image_folder_path = "./dataset/images"
    output_folder = "output"

    train_list, val_list = validate_and_create_lists(dataset_path)

    train_model(dataset_path, train_list, val_list, num_species, epochs, model_weights_path)

    # Get a list of all image files in the folder
    image_files = os.listdir(image_folder_path)
    # Randomly select 10 images
    selected_images = random.sample(image_files, k=10)

    # Visualize each selected image
    for image_name in selected_images:
        visualize(model_weights_path, image_folder_path, image_name, output_folder)
