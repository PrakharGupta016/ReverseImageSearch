from django.db import models
from django.forms import fields
from .models import ImageUpload
from django import forms

class UserImage(forms.ModelForm):
    class meta:
        # To specify the model to be used to create form
        models = ImageUpload
        # It includes all the fields of model
        fields = '__all__'

    def __str__(self):
        return self.caption