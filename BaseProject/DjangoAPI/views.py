from django.shortcuts import redirect, render
from .forms import UserImage
from .models import ImageUpload
from .ModelpythonFile import search,extract_features
def image_request(request):
    if request.method == 'POST':
        form = UserImage(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            # Getting the current instance object to display in the templates
            img_object = form.instance
            return render(request, 'index.html', {'form': form, 'img_obj': img_object})
    else:
        form = UserImage()

    return render(request, 'index.html', {'form': form})


