from django.urls import path
from . import views
from classification import views  # `views` modülünü import edin

urlpatterns = [
    path('', views.index, name='index'),
    path('cnn/', views.predict_category, name='cnn'),
    path('hybrid/', views.predict_category, name='hybrid'),
    #path('transformer/', views.predict_category, name='transformer'),
    path('lstm/', views.predict_category, name='lstm'),
]
