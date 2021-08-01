import numpy as np

from django.shortcuts import render
import sqlite3
import pandas as pd
import json
# import datatable as dt

# Create your views here.


def showdata_view(httprequest, *args, **kwargs):
#def showdata_view(dataframe):
    """Do  anything with request"""
    #df=dataframe

    conn = sqlite3.connect('TestDB1.db')

    '''
    c = conn.cursor()
    c.execute(   
        #SELECT * FROM FRAUDS)

    showData=c.fetchall()
    '''
    # Use an pandas dataframe as an import file for datatables
    df = pd.read_sql_query("SELECT * FROM FRAUDS", conn)

    """Bis hier entfernen wenn Daten direkt aus DataFrame stammen"""



    datatypesColumns = df.dtypes
    # print(datatypesColumns['Einkaufsbeleg'])  #Ausgabe in Konsole zu Testzwecken
    df = df.append(datatypesColumns, ignore_index=True)

    # One Try to commit Dataframe via global variable
    '''
    global_df = httprequest.session.get('global_df')
    if global_df is None:
        global_df = df.to_json()
    else:
        global_df = df.to_json()

    httprequest.session['global_df'] = global_df    
    '''



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


def readtable_view(httprequest):
    if httprequest.method == 'POST':
        print("Data received from Ajax ~readtable_view!")
        #dataTypesChecked = httprequest.POST['dataTypesChecked']
        dataTypesChecked = json.loads(httprequest.POST.get('dataTypesChecked'))

        print(dataTypesChecked)


        # One try to get Dataframe from global variable
        '''
        data = httprequest.session.get('global_df')
        df1= json.loads(data)
        #df = pd.json_normalize(data['results'])
        #print("Inhalt global_df:", df1)


        #df1.dtypes=dataTypesChecked
        '''

    # Reads all the displayed table on the website and saves them as a list of dataframes
    readtable = pd.read_html("http://127.0.0.1:8000/showdata/")

    # In case there could be more than 1 table displayed on the website:
    '''
    if(readtable.count()>1):
        df_read = readtable[0]
    else:
        df_read = readtable[0]
    '''

    # Creates a new Dataframe from the List of possible Dataframes
    df_read = pd.DataFrame(readtable[0])
    print("Eingelesene Tabelle inklusive Datentypen:")
    print(df_read)

    # df_optimized = df_read.drop(df_read.tail(-1).index, inplace=True)

    # Since there is a unneeded row (datatypes) in the Dataframe, the row will be dropped
    df_optimized = df_read[:-1]
    print("Optimized Dataframe:")
    print(df_optimized)

    # Compares the List of Dataframe types with the selected types from the frontend and parses them

    i = 0
    # df_completed = df_optimized
    #Parses all columns to string -> no idea why
    df_completed = df_optimized.convert_dtypes()
    print(df_completed.dtypes)

    for element in dataTypesChecked:

        if i >= len(dataTypesChecked):
            break

        print(element)
        print(i)
        spaltenname = df_completed.columns[i]
        print(spaltenname)

        if element == "INTEGER":
            print("Es handelt sich um einen Integer")
            df_completed[spaltenname] = df_completed[spaltenname].astype(np.int64)
            # df_completed[i] = df_completed[i].pd.to_numeric(df_completed[i], downcast="int64", errors='ignore')
            # df_completed[i] = df_completed.astype({df_completed[i]: 'int64'}).dtypes
            print(df_completed[spaltenname].dtypes)
            i = i+1

        if element == "FLOAT":
            print("Es handelt sich um einen Float")
            df_completed[spaltenname] = df_completed[spaltenname].astype(np.float)
            print(df_completed[spaltenname].dtypes)
            # df_completed[i] = pd.to_numeric(df_completed[i], downcast="float32", errors='ignore')
            # df_completed.columns[i] = pd.to_numeric(df_completed.columns[i], downcast='float32', errors='ignore')
            i = i+1

        if element == "STRING":
            print("Es handelt sich um einen String")
            # df_completed[spaltenname] = df_completed[spaltenname].apply(str)
            df_completed[spaltenname] = df_completed[spaltenname].astype('string')
            print(df_completed[spaltenname].dtypes)
            i = i+1

        else:
            i = i+1

    print(df_completed.dtypes)
    print("Dataframe successfully parsed!")

    # Save Dataframe to DB
    '''
        conn = sqlite3.connect('TestDB1.db')
        c = conn.cursor()
        
        # check if FRAUD table exists
        c.execute('SELECT count(name) FROM sqlite_master WHERE type='table' AND name='FRAUDS'')
        
        # if the count is 1, then table exists
        if c.fetchone()[0] == 1:
            # replace the current table with new data from dataframe
            df_completed.to_sql('FRAUDS', conn, if_exists='replace', index=False)
            conn.commit()
            
        # if table does not exist, create a new table and import data from dataframe
        else:
            c.execute('CREATE TABLE FRAUDS (Col1 text, Col2 number)')
            df_completed.to_sql('FRAUDS', conn, if_exists='replace', index=False)
            conn.commit()
        
        c.execute('SELECT * FROM FRAUDS')
        
        # close connection to database
        conn.close()
    '''

    # Remove all rows which have at least one null value
    # new_df = df.dropna(axis = 0, how = 'any', inplace = True)

    context = {
        "data": df_read.to_html(classes="display table table-striped table-hover", table_id="dataShowTable", index=False,
                           justify="center", header=True,)  # classes="table table-bordered"
    }

    # context = {"data": "Dummytext"}
    return render(httprequest, "myTemplates/showdata.html", context)
