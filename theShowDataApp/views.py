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
    print(datatypesColumns['Einkaufsbeleg'])  #Ausgabe in Konsole zu Testzwecken
    df = df.append(datatypesColumns, ignore_index=True)

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

    readtable = pd.read_html("http://127.0.0.1:8000/showdata/")
    #if(readtable.count() >1):
    df1 = readtable[0]

    #df1.dtypes=[]

    # Remove all rows which have at least one null value
    # new_df = df.dropna(axis = 0, how = 'any', inplace = True)
    print(df1)
    context = {
        "data": df1.to_html(classes="display table table-striped table-hover", table_id="dataShowTable", index=False,
                           justify="center", header=True,)  # classes="table table-bordered"
    }

    return render(httprequest, "myTemplates/showdata.html", context)
