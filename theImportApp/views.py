from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponse
from django import template
import pandas as pd


def fileimport_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    #return HttpResponse("Hello, world. You're at the polls index.")
    print("##### Inside fileimport_view")
    return render(httprequest, "myTemplates/fileimport.html")


def upload_func(httprequest):
    print()
    #pd.DataFrame()
    df= pd.read_csv("", sep=";")
    print(df)