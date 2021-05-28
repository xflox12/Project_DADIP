from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponse
from django import template


#def fileimport_view(request):
  #  html_template = loader.get_template('import.html')
   # return HttpResponse(html_template.render(request))

def fileimport_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    return render(httprequest, "import.html")