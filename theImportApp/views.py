from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from django.http import HttpResponse
from django.template import loader
from django.http import HttpRequest
from django import template
import pandas as pd
import numpy as np
import sqlite3
from theShowDataApp.views import showdata_view
from time import sleep
import pickle

# Create your views here.

def fileimport_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    print("##### Inside fileimport_view")
    return render(httprequest, "myTemplates/fileimport.html")


def upload_func(HttpRequest, *args, **kwargs):
    """ Function to handle the uploaded File.
    File will be saved to local folder "uploadStorage" in Core. The Filename will be extended with the current
    date and time so multiple uploads with the same name can be logically seperated.

    Authors: Florian, Marco
    """
    # get current Date and Time for timestamp
    _datetime = datetime.now()
    datetime_str = _datetime.strftime(
        "%Y-%m-%d_%H-%M")  # mit Uhrzeit: datetime_str = _datetime.strftime("%Y-%m-%d-%H-%M-%S")
    # Datetime for database table name
    datetime_str_pickle = _datetime.strftime("%Y_%m_%d_%H_%M")

    context = {}
    if HttpRequest.method == 'POST' and HttpRequest.FILES.get('newFile'):
        myFile = HttpRequest.FILES.get('newFile')
        fs = FileSystemStorage()

        # Split filename to add timestamp to the name and save parts of the name
        # if there are more than one dots create a list and use last part to remove extension
        file_name_split = myFile.name.split('.')
        file_name_list = file_name_split[:-1]
        ext = file_name_split[-1]
        file_name_without_ext = '.'.join(file_name_list)

        # Create new filename with timestamp
        myFile.name = file_name_without_ext + '_' + datetime_str + '.' + ext

        # Create pickle file for saving the file name as table name in database (showdata.views)
        filename_database = file_name_without_ext + '_' + datetime_str_pickle
        f = open('filename_for_database.pkl', 'wb')
        pickle.dump(filename_database, f)
        f.close()

        # save file in core/uploadStorage folder
        filename = fs.save(myFile.name, myFile)
        uploaded_file_url = fs.url(filename)

        print('########## Upload successful! ##########')
        #print(myFile)
        #print(uploaded_file_url)

        """Parse file into dataframe via pandas (without extra button)"""
        pandas_func(uploaded_file_url)

        context2 = {
            'uploaded_file_url': uploaded_file_url,
        }
        context.update(context2)

        print(context)

    else:
        print('Error. Wrong method!')
        context2 = {
            'error_message': 'No file URL existing!',
        }

        context.update(context2)

    return render(HttpRequest, "myTemplates/fileimport.html", context)



def pandas_func(filepath):
    """ Function to parse the uploaded file (remove empty columns, etc.)

    Authors: Florian, Marco
    """

    print('\n##### Start Parsing File...')

    pd.DataFrame()

    file_name_split = filepath.split('.')
    ext = file_name_split[-1]
    df = pd.DataFrame()

    # Use specific pandas read function depending on filetype
    if ext == 'csv' or ext == 'CSV':
        """If CSV-File"""
        print('CSV File!')
        df = pd.read_csv("core" + filepath, sep=";")
    elif ext == 'xlsx' or ext == 'XLSX':
        """If Excel-File"""
        print('XLSX File!')
        df = pd.read_excel("core" + filepath, engine='openpyxl')

    """How to parse via pandas: https://www.geeksforgeeks.org/drop-empty-columns-in-pandas/"""
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)  # replace all empty places with null
    df.replace(0, nan_value, inplace=True)  # replace all zeros with null

    df.dropna(how='all', axis=1, inplace=True)  # remove all null value columns

    # Save the Dataframe as pickle-File
    df.to_pickle('dataframe_before_datatype_checked.pkl')

    print('##### ... Parsing finished!\n')

