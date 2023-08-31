from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import ImageUpload
class ImageView(APIView):
    def post(self,request):
        serializer = ImageSerializer
        file = request.data['file']
        caption = request.data['caption']
        image = ImageUpload.objects.create(caption = caption,image = file)
        return Response({"status": "success", "data": caption}, status=status.HTTP_200_OK)