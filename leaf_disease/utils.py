import torch
from torchvision import transforms
from PIL import Image as PILImage
from .models import BinaryClassificationModel, MultiClassClassificationModel

# Define transformations for preprocessing images
composed_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_binary_classification_model():
    # Load the binary classification model from the saved .pth file
    model = BinaryClassificationModel.objects.first()  # Assuming there's only one model instance
    model_path = model.weights_path
    # Load the model using torch.load
    binary_model = torch.load(model_path)
    binary_model.eval()  # Set the model to evaluation mode
    return binary_model

def load_multi_class_classification_model():
    # Load the multi-class classification model from the saved .pth file
    model = MultiClassClassificationModel.objects.first()  # Assuming there's only one model instance
    model_path = model.weights_path
    # Load the model using torch.load
    multi_class_model = torch.load(model_path)
    multi_class_model.eval()  # Set the model to evaluation mode
    return multi_class_model

def predict_binary_class(image):
    # Preprocess the image
    preprocessed_image = composed_transform(image)
    # Load the binary classification model
    binary_model = load_binary_classification_model()
    # Perform inference
    with torch.no_grad():
        output = binary_model(preprocessed_image.unsqueeze(0))
    # Convert the output to probability and return the predicted class
    probability = torch.sigmoid(output).item()
    return probability >= 0.5  # Assuming 0.5 as the threshold for classification

def predict_multi_class(image):
    # Preprocess the image
    preprocessed_image = composed_transform(image)
    # Load the multi-class classification model
    multi_class_model = load_multi_class_classification_model()
    # Perform inference
    with torch.no_grad():
        output = multi_class_model(preprocessed_image.unsqueeze(0))
    # Convert the output to probabilities and return the top predicted classes
    probabilities = torch.softmax(output, dim=1).squeeze().tolist()
    # Assuming the model output is a list of probabilities corresponding to each class
    # You can further process the output to get top predicted classes
    top_classes = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
    top_diseases = {f"Disease {i+1}": probabilities[i] for i in top_classes}
    return top_diseases
