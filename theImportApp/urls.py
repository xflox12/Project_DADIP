from django.urls import path, re_path
from . import views


urlpatterns = [

    path('fileimport/', views.fileimport_view, name='fileimport'),
    path('file-upload/', views.upload_func, name='file-upload'),
    path('parse-csv-pandas/', views.pandas_func, name='parse-csv-pandas'),

]
