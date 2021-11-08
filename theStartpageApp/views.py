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
import sqlite3
import json


@login_required(login_url="/login/")
def index(request):
    """ Initial Call when starting the App. Get data from last usage.
       Authors: Marco
    """

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

    # Get Info about database
    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()
    c.execute('''SELECT COUNT(name) FROM sqlite_master WHERE type='table' ''')
    number_of_datasets = json.dumps(c.fetchall())[2:-2]
    conn.close()

    # Get Info about registered users
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    #c.execute('''SELECT COUNT(username) FROM auth_user ''')
    registered_users = 1 #json.dumps(c.fetchall())
    conn.close()



    context = {
        "last_analysis_table": last_analysis_table,
        "datetime_last_analysis": datetime_last_analysis,
        "count_fraud": count_fraud,
        "count_nonfraud": count_nonfraud,
        "registered_users": registered_users,
        "number_of_datasets": number_of_datasets,
        "segment": 'index',
    }
    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):
    """
    Call of the single pages by their URL-Name. Part of the Template.
    """
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


def faq_view(request):
    return render(request, "faq.html")


def terms_view(request):
    return render(request, "terms-conditions.html")


def legal_view(request):
    return render(request, "legal-notice.html")


def privacy_view(request):
    return render(request, "data-privacy.html")
