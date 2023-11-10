from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import ImageUpload
from .ModelpythonFile import extract_features,search
from rest_framework.response import Response
from rest_framework import pagination
from django.core.cache import cache
from rest_framework.pagination import PageNumberPagination

class CustomPagination(PageNumberPagination):
    page_size = 20  # Adjust the number of items per page as needed
    page_size_query_param = 'page_size'
    max_page_size = 100
class ImageView(APIView):
    def post(self,request):
        # serializer = ImageSerializer(data=request.data)
        # print(uploaded_image)
        file = request.data['file']
        caption = request.data['caption']
        # print(request.data)
        image = ImageUpload.objects.create(caption = caption,image = file)
        # print(image)
        return Response({"status": "success", "data": caption}, status=status.HTTP_200_OK)
class ImageSearch(APIView):


    def post(self,request):
        file = request.data['file']

        numItems = int(request.data['numItems'])
        uploadedImage = ImageUpload.objects.create(caption = "default_caption",image=file)
        image_path = uploadedImage.image.path

        result = search(extract_features(image_path),numItems);
        cache.set('store_key', result, timeout=None)

        return Response(status=status.HTTP_201_CREATED,data={"result stored"});
    def get(self,request):
        data = cache.get('store_key')
        if data is not None:
            paginator = CustomPagination()
            paginated_data = paginator.paginate_queryset(data, request)
            return paginator.get_paginated_response({"status": "success", "data": paginated_data})
        else:
            return Response({"status": "error", "message": "Data not found"}, status=status.HTTP_404_NOT_FOUND)