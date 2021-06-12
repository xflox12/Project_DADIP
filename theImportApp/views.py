from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from datetime import datetime

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.http import HttpRequest
from django import template
import pandas as pd
import numpy as np
import sqlite3


def fileimport_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    #return HttpResponse("Hello, world. You're at the polls index.")
    print("##### Inside fileimport_view")
    return render(httprequest, "myTemplates/fileimport.html")


def upload_func(HttpRequest):
    #get current Date and Time for timestamp
    _datetime = datetime.now()
    datetime_str = _datetime.strftime("%Y-%m-%d") # mit Uhrzeit: datetime_str = _datetime.strftime("%Y-%m-%d-%H-%M-%S")

    if HttpRequest.method == 'POST' and HttpRequest.FILES.get('newFile'):
        myFile = HttpRequest.FILES.get('newFile')
        fs = FileSystemStorage()

        #Split filename to add timestamp to the name and save parts of the name
        # if there are more than one dots create a list and use last part to remove extension
        file_name_split = myFile.name.split('.')
        file_name_list = file_name_split[:-1]
        ext = file_name_split[-1]
        file_name_without_ext = '.'.join(file_name_list)

        #Create new filenmae with timestamp
        myFile.name = file_name_without_ext+'_'+datetime_str+'.'+ext

        #save file in core/uploadStorage folder
        filename = fs.save(myFile.name, myFile)
        uploaded_file_url = fs.url(filename)

        context = {
            'uploaded_file_url': uploaded_file_url
        }

        print('##########Upload hat geklappt!')
        print(myFile)
        print(uploaded_file_url)

        #parse file into database via pandas
        #pandas_func(uploaded_file_url)

    else:
        print('Error. Wrong method!')
        context = {
            'error_message': 'No file URL existing!'
        }
    return render(HttpRequest, "myTemplates/fileimport.html", context)

"""Später Übergabeparameter einfügen: filepath"""
def pandas_func(HttpRequest):

    print('\n##### Start Parsing File...')

    filepath = 'core/uploadStorage/EKKO_2021-06-10.XLSX'
    #pd.DataFrame()
    #df = pd.read_csv(filepath, sep=";")
    df = pd.read_excel(filepath, engine='openpyxl')

    dummyVar = np.nan

    """How to parse via pandas: https://www.geeksforgeeks.org/drop-empty-columns-in-pandas/"""
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)     #replace all empty places with null
    df.replace(0, nan_value, inplace=True)      #replace all zeros with null

    df.dropna(how='all', axis=1, inplace=True)  # remove all null value columns


    print(df)

    print('##### ... Parsing finished!\n')

    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()

    c.execute('CREATE TABLE FRAUDS (Col1 text, Col2 number)')
    conn.commit()

    df.to_sql('FRAUDS', conn, if_exists='replace', index=False)

    c.execute('''  
    SELECT * FROM FRAUDS
              ''')

    for row in c.fetchall():
        print(row)

    return render(HttpRequest, "myTemplates/fileimport.html")


