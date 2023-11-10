from django.contrib import admin
from django.urls import path,include
from .viewClass import ImageView,ImageSearch



urlpatterns = [
    path('upload',ImageView.as_view()),
    path('search',ImageSearch.as_view()),
]
