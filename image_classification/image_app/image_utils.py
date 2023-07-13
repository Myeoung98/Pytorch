import torch
from torchvision import models, transforms


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image


def predict_image(image):
    model = models.resnet50(pretrained=True)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()
