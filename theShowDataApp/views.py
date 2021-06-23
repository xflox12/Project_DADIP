from django.shortcuts import render
import sqlite3
import pandas as pd
# import datatable as dt

# Create your views here.

def showdata_view(httprequest, *args, **kwargs):
    """Do  anything with request"""


    conn = sqlite3.connect('TestDB1.db')

    '''
    c = conn.cursor()
    c.execute(   
        #SELECT * FROM FRAUDS)

    showData=c.fetchall()
    '''
    # Use an pandas dataframe as an import file for datatables
    df = pd.read_sql_query("SELECT * FROM FRAUDS", conn)

    context = {
        'showData': df
    }
    conn.close()

    print(context)

    return render(httprequest, "myTemplates/showdata.html", context=context)