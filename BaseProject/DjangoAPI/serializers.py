from rest_framework import serializers
from .models import ImageUpload

class ImageSerializer(serializers.ModelSerializer):
        class Meta:
            model = ImageUpload
            fieds = ('__all__')
