from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import ImageUpload
from .ModelpythonFile import extract_features,search
class ImageView(APIView):
    def post(self,request):
        # serializer = ImageSerializer(data=request.data)

        # print(uploaded_image)
        file = request.data['file']
        caption = request.data['caption']
        print(request.data)
        image = ImageUpload.objects.create(caption = caption,image = file)
        print(image)
        return Response({"status": "success", "data": caption}, status=status.HTTP_200_OK)
class ImageSearch(APIView):

    def post(self,request):
        file = request.data['file']
        caption = request.data['caption']
        uploadedImage = ImageUpload.objects.create(caption=caption, image=file)
        image_path = uploadedImage.image.path

        result = search(extract_features(image_path));
        # print(result)
        return Response({"data":"dvc"})