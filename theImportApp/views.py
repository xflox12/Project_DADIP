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
    # return HttpResponse("Hello, world. You're at the polls index.")
    print("##### Inside fileimport_view")
    return render(httprequest, "myTemplates/fileimport.html")


def upload_func(HttpRequest):
    # get current Date and Time for timestamp
    _datetime = datetime.now()
    datetime_str = _datetime.strftime(
        "%Y-%m-%d_%H-%M")  # mit Uhrzeit: datetime_str = _datetime.strftime("%Y-%m-%d-%H-%M-%S")

    if HttpRequest.method == 'POST' and HttpRequest.FILES.get('newFile'):
        myFile = HttpRequest.FILES.get('newFile')
        fs = FileSystemStorage()

        # Split filename to add timestamp to the name and save parts of the name
        # if there are more than one dots create a list and use last part to remove extension
        file_name_split = myFile.name.split('.')
        file_name_list = file_name_split[:-1]
        ext = file_name_split[-1]
        file_name_without_ext = '.'.join(file_name_list)

        # Create new filenmae with timestamp
        myFile.name = file_name_without_ext + '_' + datetime_str + '.' + ext

        # save file in core/uploadStorage folder
        filename = fs.save(myFile.name, myFile)
        uploaded_file_url = fs.url(filename)

        context = {
            'uploaded_file_url': uploaded_file_url
        }

        print('##########Upload hat geklappt!')
        print(myFile)
        print(uploaded_file_url)

        # parse file into database via pandas
        # pandas_func(uploaded_file_url)

        # df = pd.read_excel("myFile", engine = 'openpyxl')



    else:
        print('Error. Wrong method!')
        context = {
            'error_message': 'No file URL existing!'
        }
    return render(HttpRequest, "myTemplates/fileimport.html", context)


"""Später Übergabeparameter einfügen: filepath"""


# def pandas_func(filepath):
def pandas_func(HttpRequest):
    print('\n##### Start Parsing File...')

    filepath = 'core/uploadStorage/EKKO_2021-06-10.XLSX'  # muss auskommentiert werden
    pd.DataFrame()
    """Für CSV-Files"""
    # df = pd.read_csv(filepath, sep=";")
    """Für Excel-Files"""
    df = pd.read_excel(filepath, engine='openpyxl')

    print(df)

    dummyVar = np.nan

    """How to parse via pandas: https://www.geeksforgeeks.org/drop-empty-columns-in-pandas/"""
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)  # replace all empty places with null
    df.replace(0, nan_value, inplace=True)  # replace all zeros with null

    df.dropna(how='all', axis=1, inplace=True)  # remove all null value columns

    # Remove all rows which have at least one null value
    # new_df = df.dropna(axis = 0, how = 'any', inplace = True)

    print(df)

    print('##### ... Parsing finished!\n')

    df.to_excel("output.xlsx", sheet_name='Parsed_Frauds_1')

    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()

    # check if FRAUD table exists
    c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='FRAUDS' ''')

    # df.to_sql('FRAUDS', conn, ='rif_existseplace', index=False)
    # if the count is 1, then table exists
    if c.fetchone()[0] == 1:

        # replace the current table with new data from dataframe
        df.to_sql('FRAUDS', conn, if_exists='replace', index=False)
        conn.commit()
    # if table does not exist, create a new table and import data from dataframe
    else:
        c.execute('CREATE TABLE FRAUDS (Col1 text, Col2 number)')
        df.to_sql('FRAUDS', conn, if_exists='replace', index=False)
        conn.commit()

    c.execute('SELECT * FROM FRAUDS')

    for row in c.fetchall():
        print(row)

    # close connection to database
    conn.close()

    return render(HttpRequest, "myTemplates/fileimport.html")


def pandas_to_sql(df, DjangoModel, if_exists="fail"):
    """Uses bulk_create to insert data to Django table
    if_exists: see pd.DataFrame.to_sql API

    Ref: https://www.webforefront.com/django/multiplemodelrecords.html
    """
    import numpy as np

    if if_exists not in ["fail", "replace", "append"]:
        raise Exception("if_exists must be fail, replace or append")

    if if_exists == "replace":
        DjangoModel.objects.all().delete()
    elif if_exists == "fail":
        if DjangoModel.objects.all().count() > 0:
            raise ValueError("Data already exists in this table")
    else:
        pass

    dct = df.replace({np.nan: None}).to_dict(
        "records"
    )  # replace NaN with None since Django doesn't understand NaN

    bulk_list = []
    for x in dct:
        bulk_list.append(DjangoModel(**x))
    DjangoModel.objects.bulk_create(bulk_list)
    print("Successfully saved DataFrame to Django table.")

# return render(HttpRequest, "myTemplates/fileimport.html")
