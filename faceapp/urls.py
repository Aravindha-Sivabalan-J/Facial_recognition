from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.login_view, name="login_view"),
    path('home/', views.home_view, name="home_view"),
    path('logout/', views.logout_view, name="logout"),
    path('register/', views.register_view, name="register_view"),
    path('addprofile/', views.create_profile, name="create_profile"),
    path('searchdbpage/', views.searchdb_view,name="searchdbpage"),
    path('compareurl/', views.compare_faces, name="comparefaces"),
    path('videoscanning/', views.video_scanner, name = "videoscanning"),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)