from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home_view_temp, name='home'),
]