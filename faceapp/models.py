from django.db import models

# Create your models here.
class Profile(models.Model):
    name = models.CharField(max_length=35, default='-NA-')
    photo = models.ImageField(blank=True, upload_to='photos')
    age = models.IntegerField(null=False, blank=False, default=000)
    bio = models.TextField()
    Facenet512_embedding = models.BinaryField(blank=True, null=True)
    Facenet_embedding = models.BinaryField(blank=True, null=True)
    Dlib_embedding = models.BinaryField(blank=True, null=True)
    VGGFace_embedding = models.BinaryField(blank=True, null=True)
    ArcFace_embedding = models.BinaryField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
