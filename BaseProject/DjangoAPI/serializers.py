from rest_framework import serializers
from .models import ImageUpload

class ImageSerializer(serializers.ModelSerializer):
        caption = serializers.CharField(max_length=200)
        image = serializers.ImageField()
        class Meta:
            model = ImageUpload
            fieds = ('__all__')
