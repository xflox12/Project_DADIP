from django.urls import path, re_path
from . import views


urlpatterns = [

    # The home page
    path('fileimport/', views.fileimport_view, name='fileimport'),

]
