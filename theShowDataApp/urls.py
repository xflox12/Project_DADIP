from django.urls import path, re_path
from . import views


urlpatterns = [

    path('showdata/', views.showdata_view, name='showdata'),
    path('readtable/', views.readtable_view, name='readtable'),

]