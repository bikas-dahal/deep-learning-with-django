
from django.shortcuts import render
# from .models import BinaryClassificationModel, MultiClassClassificationModel
from .forms import ImageUploadForm
from .utils import predict_binary_class 

from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadFileForm
from .predict import predict_image,  classify_top_diseases

def check(request):
    return render(request, 'leaf_disease_detection/check.html')



def mix(request):
    prediction_message = None
    top_diseases = None
    
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # Perform binary classification
            is_leaf = predict_image(image)
            if is_leaf:
                # prediction_message = "We process your provided ☘️ leaf image:🙌"
                # Perform disease classification
                top_diseases = classify_top_diseases(image)
            else:
                prediction_message = "Given input is not an image of a leaf❌, please try again.👋"
    else:
        form = UploadFileForm()
    
    return render(request, 'mix.html', {'form': form, 'prediction_message': prediction_message, 'top_diseases': top_diseases})



def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            prediction = predict_image(image)
            prediction = int(prediction)
            prediction = prediction + 1
            print(prediction)
            print(type(prediction))
            message = ""
            if prediction == 1:
                message = "Your given image is not a plant leaf❌, please try again.👋"
            elif prediction == 2:
                message = "Your provided image is a ☘️ plant leaf👏"
            else:
                message = "Prediction is not available."
            return render(request, 'index.html', {'prediction': prediction, 'message': message})
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})


def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            top_diseases = classify_top_diseases(image)
            return render(request, 'classify_image_result.html', {'top_diseases': top_diseases})
    else:
        form = ImageUploadForm()
    return render(request, 'classify_image.html', {'form': form})


# def index2(request):
#     if request.method == 'POST':
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             image = form.cleaned_data['image']
#             prediction = predict_image2(image)
#             # prediction = int(prediction)
#             # prediction = prediction + 1
#             # print(prediction)
#             # print(type(prediction))
#             message = ""
#             if prediction == 1:
#                 message = "Your given image is not a plant leaf, please try again."
#             elif prediction == 2:
#                 message = "Your provided image is a plant leaf."
#             else:
#                 message = "Prediction is not available."
#             return render(request, 'index.html', {'prediction': prediction, 'message': message})
#     else:
#         form = UploadFileForm()
#     return render(request, 'index.html', {'form': form})


# def process_image(request):
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             image_instance = form.save()
#             # Predict if image is a plant leaf
#             is_plant_leaf = predict_binary_class(image_instance.image)
#             if is_plant_leaf:
#                 # If image is a plant leaf, predict diseases
#                 top_diseases = predict_multi_class(image_instance.image)
#                 return render(request, 'leaf_disease_detection/process_image.html', {'top_diseases': top_diseases})
#             else:
#                 # If image is not a plant leaf, display message
#                 message = "Provide valid input: Not a plant leaf"
#                 return render(request, 'leaf_disease_detection/process_image.html', {'message': message})
#     else:
#         form = ImageUploadForm()
#     return render(request, 'leaf_disease_detection/process_image.html', {'form': form})
