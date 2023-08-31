from django.db import models
from django.forms import fields
from .models import ImageUpload
from django import forms

class UserImage(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = '__all__'

    def __str__(self):
        return self.caption