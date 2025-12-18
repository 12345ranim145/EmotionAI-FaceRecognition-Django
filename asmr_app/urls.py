from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('recognize/', views.recognize_face, name='recognize'),
    path('asmr/', views.asmr_page, name='asmr'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('generate_audio/<str:emotion>/', views.generate_audio, name='generate_audio'),
]