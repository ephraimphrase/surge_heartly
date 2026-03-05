from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.heart_attack_prediction, name='heart_attack_prediction'),
]