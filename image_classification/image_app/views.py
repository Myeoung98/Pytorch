from django.shortcuts import render
from .image_serializer import ImageUploadForm
from PIL import Image as PILImage
import torch
from torchvision import models, transforms


def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()

            # Load the pre-trained ResNet-50 model
            model = models.resnet50(pretrained=True)
            model.eval()

            # Preprocess the input image
            image_path = image.image.path
            image_tensor = preprocess_image(image_path)

            # Perform image classification
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0))
                _, predicted_idx = torch.max(output, 1)
                predicted_class = get_class_name(predicted_idx.item())

            # Update the predicted class and confidence in the database
            image.predicted_class = predicted_class
            image.confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()
            image.save()

            return render(request, 'result.html', {'image': image})

    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})


def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = PILImage.open(image_path)
    image_tensor = preprocess(image)
    return image_tensor


def get_class_name(class_idx):
    # Provide a dictionary mapping class indices to class names
    class_names = class_names = {
        0: 'class1',
        1: 'class2',
        2: 'class3',
    }
    return class_names.get(class_idx, 'Unknown')
