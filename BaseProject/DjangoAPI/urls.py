from django.contrib import admin
from django.urls import path,include
from .viewClass import ImageView



urlpatterns = [
    path('upload',ImageView.as_view())
]
