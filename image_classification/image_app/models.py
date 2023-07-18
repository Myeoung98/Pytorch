from django.db import models


def image_upload_path(instance, filename):
    return f'images/{filename}'


class Image(models.Model):
    image = models.ImageField(upload_to=image_upload_path)
    predicted_class = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(blank=True, null=True)
