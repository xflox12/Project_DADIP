# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin
from django.urls import path, include  # add this

urlpatterns = [
    path('admin/', admin.site.urls),          # Django admin route
    path("", include("theAuthenticationApp.urls")), # Auth routes - login / register
    path("", include("theShowDataApp.urls")),  #Anzeige der vorhandene Daten
    path("", include("theDataVisualApp.urls")), # Visualisierung der Frauds
    path("", include('theImportApp.urls')),  # Einlesen der Datensätze
    path("", include("theMLearningApp.urls")),  # Machine Learning Algorithm

    #theStartpageApp muss als letztes eingebunden werden
    path("", include("theStartpageApp.urls")),             # UI Kits Html files

]
