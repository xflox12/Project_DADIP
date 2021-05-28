from django.shortcuts import render

# Create your views here.

def mlalgo_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    return render(httprequest, "myTemplates/page_blank.html")