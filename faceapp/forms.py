from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile

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
    