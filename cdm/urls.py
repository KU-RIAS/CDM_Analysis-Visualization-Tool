"""cdm URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.http import HttpResponse
from django.contrib import admin
from django.urls import path
from django.views.generic.base import RedirectView
import www.views as views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.indexD, name='indexD'),    
    
    path('indexA/', views.indexA, name='indexA'),    
    path('get_choiceA/', views.get_choiceA, name='choiceA'),

    path('indexB/', views.indexB, name='indexB'),
    path('get_choiceB/', views.get_choiceB, name='choiceB'),

    path('indexC/', views.indexC, name='indexC'),    
    path('get_choiceC/', views.get_choiceC, name='choiceC'),
    
    path('indexD/', views.indexD, name='indexD'),
    
    path('indexE/', views.indexE, name='indexE'),
    path('get_choiceE/', views.get_choiceE, name='choiceE'),
]
