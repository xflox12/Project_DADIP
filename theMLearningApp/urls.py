from django.urls import path, re_path
from . import views


urlpatterns = [

    path('mlalgo/', views.mlalgo_view, name='mlalgo'),
    path('find_datatables/', views.find_datatables, name='find_datatables'),

    # Test for advanced ML: ########################################
    path('train_mlalgo/', views.train_mlalgo, name='train_mlalgo'),
    # Path for Analysing without model training:
    path('analyze_file', views.analyze_file, name='analyze_file'),
]
