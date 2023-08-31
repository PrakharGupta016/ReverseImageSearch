from rest_framework import serializers
from .models import ImageUpload

class ImageSerializer(serializers.ModelSerializer):
        caption = models.CharField(max_length=200)
        image = models.ImageField(upload_to='images')
        class Meta:
            model = ImageUpload
            fieds = ('__all__')
