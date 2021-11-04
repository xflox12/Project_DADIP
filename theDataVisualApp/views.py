from django.shortcuts import render
#views.py Upload Funktion (Schritt 2)
from collections import Counter
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from django.contrib import messages
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.io as pio
from pandas import DataFrame
import numpy as np
import pickle

# Create your views here.

def datavisu_view(request):

    df_for_visu = pd.read_pickle('dataframe_encoded_and_normalized.pkl')
    print('Dataframe for Visualisation:')
    print(df_for_visu)

    # Read amount of frauds and non-frauds resulting from ML###############
    f = open('count_fraud.pkl', 'rb')
    count_fraud = pickle.load(f)
    f.close()
    file = open('count_nonfraud.pkl', 'rb')
    count_nonfraud = pickle.load(file)
    file.close()

    #box_bestellnettowert= df_for_visu["Bestellnettowert"].to_numpy()
    box_bestellnettowert = df_for_visu["Bestellnettowert"].to_json()
    box_bestellmenge= df_for_visu["Bestellmenge"].to_json()
    box_bestellnettopreis= df_for_visu["Bestellnettopreis"].to_json()

    print(box_bestellnettowert)
    #######################################################################
    context = {
        'count_fraud': count_fraud,
        'count_nonfraud': count_nonfraud,
        'bestellnettowert': box_bestellnettowert,
        'bestellmenge' : box_bestellmenge,
        'bestellnetopreis' : box_bestellnettopreis,

    }
    return render(request, "myTemplates/data-visualization.html", context)


def readfile(filename):
    #we have to create those in order to be able to access it around
    # use panda to read the file because i can use DATAFRAME to read the file
    #column;culumn2;column
    global rows,columns,data,my_file,missing_values
     #read the missing data - checking if there is a null
    missingvalue = ['?', '0', '--']
    my_file = pd.read_csv(filename, sep='[:;,|_]',na_values=missingvalue, engine='python')
    data = pd.DataFrame(data=my_file, index=None)
    print(data)
    rows = len(data.axes[0])
    columns = len(data.axes[1])

    null_data = data[data.isnull().any(axis=1)] # find where is the missing data #na null =['x1','x13']
    missing_values = len(null_data)
def results(request):
    # prepare the visualization
                                #12
    message = 'I found ' + str(rows) + ' rows and ' + str(columns) + ' columns. Missing data: ' + str(missing_values)
    messages.warning(request, message)
    dashboard = [] # ['A11','A11',A'122',]
    for x in data[attribute]:
        dashboard.append(x)
    my_dashboard = dict(Counter(dashboard)) #{'A121': 282, 'A122': 232, 'A124': 154, 'A123': 332}
    print(my_dashboard)
    keys = my_dashboard.keys() # {'A121', 'A122', 'A124', 'A123'}
    values = my_dashboard.values()
    listkeys = []
    listvalues = []
    for x in keys:
        listkeys.append(x)
    for y in values:
        listvalues.append(y)
    print(listkeys)
    print(listvalues)
    context = {
        'listkeys': listkeys,
        'listvalues': listvalues,
    }
    return render(request, "myTemplates/data-visualization.html", context)

    """ Piechart: Visualize Fraud and No Fraud 
        Authors: Julia, Sophie
        Boxplot: Visualize the columns Bestellnettowert, Bestellmenge and Bestellnettopreis
        Authors: Julia, Sophie, Florian
    """


