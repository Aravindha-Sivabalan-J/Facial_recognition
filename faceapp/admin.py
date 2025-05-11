from django.contrib import admin

# Register your models here.

from .models import Profile

class ProfileAdmin(admin.ModelAdmin):
    list_display = ('name', 'photo', 'age', 'bio', 'Facenet512_embedding', 'Facenet_embedding', 'Dlib_embedding', 'VGGFace_embedding', 'ArcFace_embedding')

admin.site.register(Profile, ProfileAdmin)