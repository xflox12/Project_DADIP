from django.urls import path, re_path
from . import views


urlpatterns = [

    path('datavisu/', views.datavisu_view, name='datavisu'),

]
