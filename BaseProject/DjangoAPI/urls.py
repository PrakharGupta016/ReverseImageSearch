from django.contrib import admin
from django.urls import path,include



from .views import image_request

urlpatterns = [
    path('', image_request, name = "image-request")
]
