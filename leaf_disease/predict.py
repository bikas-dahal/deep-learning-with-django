import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models/resnet_binary.pth')



import os
import torch
from torchvision import models, transforms
from PIL import Image

def predict_image(image):
    # Load the model architecture
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming you have 2 output classes

    # Load the state dictionary from the .pth file
    model_filename = 'resnet_binary.pth'
    model_path = os.path.join(os.path.dirname(__file__), 'models', model_filename)
    state_dict = torch.load(model_path)

    # Assign the state dictionary to the model
    model.load_state_dict(state_dict)

    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define the image transformation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    image = Image.open(image)
    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Move image tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Return prediction result
    return predicted.item()  # Assuming binary classification, adjust as needed


import os
import torch
import torchvision.transforms as transforms
from PIL import Image
# from .models import DiseaseClassificationModel
# from .utils import predict_disease


import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np



import torch
import torch.nn as nn

# Define BasicBlock for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if the input and output dimensions do not match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

# Define the ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes=13):  # Set num_classes to 13
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64  # Initialize in_channels
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# # Example usage:
# model = ResNet18(num_classes=13).to(device)  # Specify 13 classes for classification
# example_input = torch.randn((1, 3, 224, 224)).to(device)  # Example input tensor
# output = resnet18_model(example_input)
# print(output.shape)  # Check the shape of the output





def classify_top_diseases(image):
    # Load the model
    model = ResNet18(num_classes=13)  # Assuming you have a model class defined
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 13)  # Assuming num_classes is defined

    # Load the model weights
    model_filename = 'resnet_18.pth'
    model_filename = 'resnet_18_20_epoch.pth'

    model_path = os.path.join(os.path.dirname(__file__), 'models', model_filename)
    # model_path = 'path_to_model.pth'  # Adjust the path to your trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    img = Image.open(image)
    img = transform(img).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top 3 predictions and their corresponding probabilities
    class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]

    
    top_probabilities, top_indices = torch.topk(probabilities, 3)
    top_diseases = [(class_names[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probabilities)]

    return top_diseases


# def classify_disease(image):
#     # Load the pre-trained ResNet18 model
#     model = DiseaseClassificationModel()  # Assuming you have a model class defined
    
#     model = models.resnet18(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = torch.nn.Linear(num_ftrs, 13)
    
#     # Load the trained model weights
#     model_filename = 'resnet_18_20_epoch.pth'
#     model_path = os.path.join(os.path.dirname(__file__), 'models', model_filename)
#     model.load_state_dict(torch.load(model_path))
    
#     # Preprocess the image
#     # preprocess = transforms.Compose([
#     #     transforms.Resize(256),
#     #     transforms.CenterCrop(224),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     # ])
    
    # preprocess = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
#     image = preprocess(image)
#     image = image.unsqueeze(0)  # Add batch dimension
    
#     # Make predictions
#     model.eval()
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
    
#     # Map predicted index to disease name
#     classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
#                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
#                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
#     predicted_class = classes[predicted.item()]
    
#     return predicted_class



# def predict_image(image):
#     # Define the model architecture
#     model = models.resnet18(pretrained=False)


#     num_ftrs = model.fc.in_features
#     model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming you have 2 output classes

#     # Load the state dictionary from the .pth file
#     model_filename = 'resnet_binary.pth'
#     model_path = os.path.join(os.path.dirname(__file__), 'models', model_filename)

#     state_dict = torch.load(model_path)

#     # Assign the state dictionary to the model
#     model.load_state_dict(state_dict)


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
    
#     test_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Preprocess the image
#     image_tensor = ToTensor()(Image.open(image))
#     image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         _, predicted = torch.max(outputs, 1)
    
#     # Return prediction result
#     return predicted.item()  # Assuming binary classification, adjust as needed
