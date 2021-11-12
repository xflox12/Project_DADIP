# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
#from theStartpageApp import views   #alternative zum einbinden
from . import views

urlpatterns = [

# Path to Indexpage
    path('index/', views.index, name='index'),

    # Path to Frequently Asked Questions
    path('faq/', views.faq_view, name='faq'),

    # Path to Terms and Conditions
    path('terms/', views.terms_view, name='terms'),

    # Path to Legal Note
    path('legal/', views.legal_view, name='legal'),

    # Path to Data Privacy
    path('privacy/', views.privacy_view, name='privacy'),

    # The home page
    path('', views.index, name='home'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
