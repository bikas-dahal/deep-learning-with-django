from django import forms
from .models import Image

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']
        
        
from django import forms

class UploadFileForm(forms.Form):
    image = forms.ImageField()


class ImageUploadForm(forms.Form):
    image = forms.ImageField()

