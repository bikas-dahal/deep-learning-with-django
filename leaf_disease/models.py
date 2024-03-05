
from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images/')

class BinaryClassificationModel(models.Model):
    # Define fields for the binary classification model
    # For example, if your model has weights and biases stored in a .pth file,
    # you might define a field to store the path to the .pth file.
    weights_path = models.CharField(max_length=255)

class MultiClassClassificationModel(models.Model):
    # Define fields for the multi-class classification model
    weights_path = models.CharField(max_length=255)
