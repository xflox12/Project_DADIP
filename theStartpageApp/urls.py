# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
#from theStartpageApp import views   #alternative zum einbinden
from . import views

urlpatterns = [

    path('faq/', views.faq_view, name='faq'),

    # The home page
    path('', views.index, name='home'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
