# Object Detection Training Pipeline
This project aims to design and implement a training pipeline for an object detection model using Python and PyTorch. The model is designed to detect one object in each image and output various attributes for the detected object. Specifically, the model will determine whether an object is present in the image, the bounding box coordinates of the detected object, whether the object is a cat or a dog, and the specific species of the detected object.

## Dataset Description
The dataset consists of 1041 images, each containing a single object, either a cat or a dog. There are a total of 9 species for both cats and dogs. The distribution of images in each species category is as follows:

**Cat** [3 species]:

Abyssinian: 92 images
Birman: 93 images
Persian: 93 images
**Dog** [6 species]:

american_bulldog: 93 images
american_pit_bull_terrier: 93 images
basset_hound: 93 images
beagle: 93 images
chihuahua: 93 images
pomeranian: 93 images
Images without a cat or dog (empty): 142 images

The dataset is provided in two folders:

**images**: Contains 986 images in JPEG format.
**labels**: Contains 899 XML files, each corresponding to an image and providing details about the image labels. Images without cats or dogs do not have a corresponding XML file.
### Project Tasks
#### Model Design
Choose a suitable backbone for the object detection model, which can be sourced from PyTorch (torchvision) or any other reputable online resource.
Design a custom head for the model that can handle the multiple outputs required: object presence, bounding box coordinates, cat/dog identification, and species classification.
#### Custom Dataloader
Implement a custom dataloader to handle the unique format of the dataset.
Apply preprocessing and augmentations to enhance the model's training performance.
Loss Function
Design and implement a loss function that can handle all of the required outputs of the model. PyTorch built-in loss functions can be used.
#### Test Function
Develop a test function to evaluate the model on the validation set and calculate metrics for all the required outputs.
Resume Training Functionality
Implement functionality to resume training from the exact point where it was stopped if model weights file is provided.
#### Visualize Function
Create a visualize function that takes the path of a folder with images and the weight file as input and returns a dictionary of dictionaries with the model's output for each image.
The output should include whether an object is present, cat/dog identification, species classification, and bounding box coordinates if an object is detected.
The function should save the output images with bounding boxes drawn on them.
