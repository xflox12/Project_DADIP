# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
import pickle


@login_required(login_url="/login/")
def index(request):

    # Try if files exist, else dummy output

    # Info about name of last Table analysed
    try:
        f = open('last_analysis_table.pkl', 'rb')
        last_analysis_table = pickle.load(f)
        f.close()
    except:
        last_analysis_table = ""

    # Info about the date of analysis
    try:
        f = open('datetime_last_analysis.pkl', 'rb')
        datetime_last_analysis = pickle.load(f)
        f.close()
    except:
            datetime_last_analysis = "No File analysed before..."

    # Info about the amount of fraud detected
    try:
        f = open('count_fraud.pkl', 'rb')
        count_fraud = pickle.load(f)
        f.close()
    except:
        count_fraud = ""

    # Info about the amount of non-fraud detected
    try:
        f = open('count_nonfraud.pkl', 'rb')
        count_nonfraud = pickle.load(f)
        f.close()
    except:
        count_nonfraud = ""

    context = {
        "last_analysis_table": last_analysis_table,
        "datetime_last_analysis": datetime_last_analysis,
        "count_fraud": count_fraud,
        "count_nonfraud": count_nonfraud,
        "segment": 'index',
    }
    # For Standard: Just use following Code
    # context = {}
    # context['segment'] = 'index'

    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))
        
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))
