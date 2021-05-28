from django.urls import path, re_path
from . import views


urlpatterns = [

    path('mlalgo/', views.mlalgo_view, name='mlalgo'),

]
