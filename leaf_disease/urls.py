from django.urls import path
from . import views

urlpatterns = [
    path('process_image/', views.classify_image, name='process_image'),
    path('', views.check, name='check'),
    path('check/', views.index, name='index'),
    # path('check2', views.index2, name='index2'),
    path('mix', views.mix, name='mix'),
]

