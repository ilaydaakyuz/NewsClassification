from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('cnn/', views.cnn, name='cnn'),
    path('hybrid/', views.hybrid, name='hybrid'),
    path('transformer/', views.transformer, name='transformer'),
    path('lstm/', views.lstm, name='lstm'),
]
