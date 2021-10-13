from django.urls import path, re_path
from . import views


urlpatterns = [

    # Return View
    path('mlalgo/', views.mlalgo_view, name='mlalgo'),

    # Start function to read all tables in Database and return them to frontend
    path('find_datatables/', views.find_datatables, name='find_datatables'),

    # Start function to train ML Model and Analyse Test Data
    path('train_mlalgo/', views.train_mlalgo, name='train_mlalgo'),

    # Function for analysing without model training:
    path('analyze_file', views.analyze_file, name='analyze_file'),
]
