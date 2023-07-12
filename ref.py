import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


# @title
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 5)  # 4 bbox points, 1 specie class

        # ### Initialize the required Layers
        # self.have_object = None
        # self.cat_or_dog = None
        # self.specie = None
        # self.bbox = None
        # ### Initialize the required Layers

    def forward(self, input):
        # backbone
        out_backbone = self.backbone(input)
        out_backbone = out_backbone.to(torch.uint8)

        # heads
        x = nn.ReLU()(self.fc1(out_backbone))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        logits = nn.Softmax()(self.output(x))

        return logits

        # return {
        #     "bbox": None,
        #     "object": None,
        #     "cat_or_dog": None,
        #     "specie": None
        # }


class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(25088, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 5)  # 4 bbox points, 1 specie class

        ### Initialize the required Layers
        self.specie = 0
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.xmax = 0

    def forward(self, input):
        # backbone
        out_backbone = self.backbone(input)
        out_backbone = out_backbone.to(torch.uint8)

        # flatten
        x = self.flat(out_backbone)

        # heads
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        pred = nn.Softmax()(self.output(x))

        return pred

    def training_step(self, batch, batch_idx):
        image, labels = batch
        predicted_labels = self(image)

        loss_specie = nn.CrossEntropyLoss()(predicted_labels[:, -1], labels[:, -1])
        loss_bbox = nn.MSELoss()(predicted_labels[:, :-1], labels[:, :-1])

        loss = torch.sum(torch.stack([loss_specie, loss_bbox]))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, labels = batch
        predicted_labels = self(image)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


# CUSTOM DATALOADER IMPLEMENTATION
train_list = np.load('./dataset/train_list.npy', allow_pickle=True).tolist()
val_list = np.load('./dataset/val_list.npy', allow_pickle=True).tolist()
from bs4 import BeautifulSoup


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


read_xml_file('dataset/labels/pomeranian_180.xml')
test = read_xml_file('dataset/labels/pomeranian_180.xml')
[float(test['xmin'])]

# list all species for class mapping
print(list(set(['_'.join(i.split('_')[:-1]) for i in os.listdir('dataset/labels/')])))
from PIL import Image

import torchvision.transforms as transforms


class CustomDataset():

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


# training_dataset = CustomDataset("dataset/", images_list=train_list)
# training_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)
# TEST IMAGE
# from google.colab.patches import cv2_imshow

# path = "dataset/images/n01877812_wallaby.jpeg"
# img = cv2.imread(path)
# cv2_imshow(img)
from torch import utils

training_dataset = CustomDataset("dataset/", images_list=train_list)
train_dataloaders = utils.data.DataLoader(training_dataset, batch_size=16, shuffle=True)

val_dataset = CustomDataset("dataset/", images_list=val_list)
val_dataloaders = utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
# # TEST DATASET AND DATALOADER
# for imgs, labels in train_dataloaders:
#     print("Batch of images has shape: ", imgs.shape)
#     print("Batch of images has type: ", type(imgs))
#     print("Batch of labels has shape: ", labels.shape)
# TRAINING LOOP IMPLEMENTATION
# Instantiate model object / Load Data
model = Model()

trainer = pl.Trainer(max_epochs=2)  # devices=1, accelerator="gpu",
trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
