from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile, Comparison_Images
from django.forms.widgets import FileInput

class ProCreationForm(UserCreationForm):
    image = forms.ImageField()
    email = forms.EmailField()
    first_name = forms.CharField(max_length=30)
    last_name = forms.CharField(max_length=30)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'image', 'email', 'password1', 'password2')

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile 
        fields = ('name', 'photo', 'age', 'bio')

class CompareFacesForm(forms.Form):
    images = forms.FileField(
        label="Upload Images",
        required=True,
        help_text = 'upload an even number of images(at least two)'
    )
    
class UploadForm(forms.ModelForm):
    class Meta:
        model = Comparison_Images
        fields = '__all__'

class MultiUploadForm(forms.Form):
    images_to_be_uploaded = forms.FileField(
        label="SELECT IMAGES TO BE UPLOADED",
        required=True,
    )