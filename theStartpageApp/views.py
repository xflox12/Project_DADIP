from django.shortcuts import render, redirect
from django.http import HttpResponse

# Create your views here.


def index(request):
    return HttpResponse("Hello, world. You're at the StartpageApp index.")

def index2(request):
    return

def home_view_temp(httprequest, *args, **kwargs):             # view with template
    """The home_view_temp is the main function which will be called when you are going to the landing page (/home).
    Before rendering the HTML-Template there will be some SQL-Querys to get the desired data from the database
    """
    return render(httprequest, "home.html")

def nextSide_view_temp(httprequest, *args, **kwargs):
    """Do  anything with request"""
    """Save data to database"""
    return render(httprequest, "seite2.html")