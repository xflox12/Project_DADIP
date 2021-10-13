import numpy as np

from django.shortcuts import render
import sqlite3
import pandas as pd
import json
import pickle


def showdata_view(HttpRequest, *args, **kwargs):
    """ Display the Dataframe that was recently created by theImportApp with an additional row for the column datatypes
    Authors: Florian, Marco
    """

    # Read the dataframe that was recently uploaded
    df = pd.read_pickle('dataframe_before_datatype_checked.pkl')  # reload created dataframe
    print('$$ showdata_view: Read pickle-File...')
    print(df)
    """
    #leave for documentation  --> first try read data via database instead of dataframe (pickle file)
    
    conn = sqlite3.connect('TestDB1.db')

    '''
    c = conn.cursor()
    c.execute(   
        #SELECT * FROM FRAUDS)

    showData=c.fetchall()
    '''
    # Use an pandas dataframe as an import file for datatables
    df = pd.read_sql_query("SELECT * FROM FRAUDS", conn)
    """

    """Bis hier entfernen wenn Daten direkt aus DataFrame stammen"""

    datatypesColumns = df.dtypes
    df = df.append(datatypesColumns, ignore_index=True)

    context = {
        "dataTypesColumns" : datatypesColumns,
        "data": df.to_html(classes="display table table-striped table-hover", table_id="dataShowTable", index=False,
                           justify="center", header=True,)
    }

    return render(HttpRequest, "myTemplates/showdata.html", context)


def readtable_view(httprequest):
    """ Function to receive the selected datatype from frontend and parses the Dataframe accordingly

    Authors: Marco, Florian
    """
    if httprequest.method == 'POST':
        print("Data received from Ajax ~readtable_view!")
        #dataTypesChecked = httprequest.POST['dataTypesChecked']
        dataTypesChecked = json.loads(httprequest.POST.get('dataTypesChecked'))

        print(dataTypesChecked)

        df_read = pd.read_pickle('dataframe_before_datatype_checked.pkl')  # reload created dataframe

        # One try to get Dataframe from global variable -> Keep for documentation
        '''
        data = httprequest.session.get('global_df')
        df1= json.loads(data)
        #df = pd.json_normalize(data['results'])
        #print("Inhalt global_df:", df1)


        #df1.dtypes=dataTypesChecked
        '''
    """
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

    # Failed experiment to delete datatype row -> can be deleted
    # df_optimized = df_read.drop(df_read.tail(-1).index, inplace=True)
    """
    # Since there is a unneeded row (datatype) in the Dataframe, the row will be dropped -> not needed since df is
    # reloaded from pickle file without the additional datatype row
    #df_optimized = df_read[:-1]
    #print("Optimized Dataframe:")
    #print(df_optimized)

    # Compares the List of Dataframe types with the selected types from the frontend and parses them

    i = 0  # count variable to prevent index errors

    # Function to convert data types automatically
    df_completed = df_read.infer_objects()
    # df_completed = pd.DataFrame(df_optimized).infer_objects()

    for element in dataTypesChecked:

        if i >= len(dataTypesChecked):
            break

        c_name = df_completed.columns[i]

        if element == "INTEGER":
            df_completed[c_name] = df_completed[c_name].astype(np.int64)
            # df_completed[i] = df_completed[i].pd.to_numeric(df_completed[i], downcast="int64", errors='ignore')
            print(c_name + " was converted to:")
            print(df_completed[c_name].dtypes)
            i = i+1

        elif element == "FLOAT":
            df_completed[c_name] = df_completed[c_name].astype(np.float)
            print(c_name + " was converted to:")
            print(df_completed[c_name].dtype)
            i = i+1

        elif element == "STRING":
            df_completed[c_name] = df_completed[c_name].astype('string')
            print(c_name + " was converted to:")
            print(df_completed[c_name].dtypes)
            i = i+1

        else:
            i = i+1

    print(df_completed.dtypes)
    print("Dataframe successfully parsed!")

    # Save Dataframe to DB
    pandas_to_sql(df_completed)
    print("Dataframe safed to SQL database")

    # Update the Pickle-file of Dataframe
    df_completed.to_pickle('dataframe_before_datatype_checked.pkl')

    context = {
        "data": df_completed.to_html(classes="display table table-striped table-hover", table_id="dataShowTable", index=False,
                           justify="center", header=True,)
    }

    return render(httprequest, "myTemplates/showdata.html", context)


def pandas_to_sql(df_completed):
    """ Function to save Dataframe to DB

    Authors: Marco
    """

    # Start Connection to DB
    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()

    # Read name from last uploaded file
    f = open('filename_for_database.pickle', 'rb')
    filename_database = pickle.load(f)
    f.close()

    # Create new table if name does not exist using the imported file name
    c.execute('CREATE TABLE IF NOT EXISTS {}(Col1 text, Col2 number)'.format("" + filename_database + ""))
    # Save dataframe to the database table
    df_completed.to_sql(filename_database, conn, if_exists='replace')  # index is true by default

    conn.commit()
    print("Table has been successfully added to Database")
    # close connection to database
    conn.close()
