from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponse
from django import template


def fileimport_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    #return HttpResponse("Hello, world. You're at the polls index.")
    print("##### Inside fileimport_view")
    return render(httprequest, "myTemplates/fileimport.html")