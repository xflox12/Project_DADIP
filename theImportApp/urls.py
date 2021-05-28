from django.urls import path, re_path
from .views import fileimport_view


urlpatterns = [

    # The home page
    path('fileimport/', fileimport_view, name='fileimport'),



]
