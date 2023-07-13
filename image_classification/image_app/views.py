from django.shortcuts import render
from .image_utils import process_image, predict_image
from django.views.decorators.csrf import ensure_csrf_cookie


@ensure_csrf_cookie
def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        processed_image = process_image(image)
        predicted_label = predict_image(processed_image)
        context = {'predicted_label': predicted_label}
        return render(request, 'result.html', context)
    return render(request, 'index.html')
