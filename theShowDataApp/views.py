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

    datatypesColumns = df.dtypes

    context = {
        "showData": df,
        "test": "Hello World",
        "dataTypesColumns" : datatypesColumns,
        # edit datatable
        "data": df.to_html(classes="display table table-striped table-hover", table_id="dataShowTable", index=False,
                           justify="center", header=True,)  # classes="table table-bordered"
    }
    conn.close()

    #print(context)

    return render(httprequest, "myTemplates/showdata.html", context)


def readtable_view(httprequest, *args, **kwargs):
    readtable = pd.read_html("myTemplates/showdata.html")
    #if(readtable.count() >1):
    df1 = readtable[0]

    return render(httprequest, "myTemplates/showdata.html")
