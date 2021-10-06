from django.shortcuts import render

# imports from google colab
import sys
import matplotlib
import scipy
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import pickle
import sqlite3

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix#
from pyod.models.knn import KNN #test 13.08.
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

# Create your views here.

def mlalgo_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    # GET ALL TABLES FROM DATABASE (for frontend dropdown selection)
    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()
    c.execute('''SELECT name FROM sqlite_master WHERE type='table' ''')
    datatable_names = c.fetchall()
    conn.close()

    #############################################################################################
    #NEW CODE (all new code within "#" lines) ###################################################
    context = {
        "datatable_names": datatable_names
    }
    return render(httprequest, "myTemplates/machine-learning.html", context)

def mlalgo_start(httprequest, *args, **kwargs):
    # takes the selection from frontend and saves it to selected_option
    if httprequest.POST:  # If this is true, the view received POST
        selected_option = httprequest.POST.get('select_df', None)
        # print("Der ausgewählte DF ist: "+selected_option)

    # Remove unnecessary charakters from selected option string
    selected_table = selected_option[2:-3]
    print("################################################################################")
    print(selected_table)

    # Read Dataframe from Database
    conn = sqlite3.connect('TestDB1.db')
    dataframe = pd.read_sql('SELECT * FROM {}'.format(""+selected_table+""), conn)
    print("The selected Dataframe from Frontend is:")
    print(dataframe)
    #############################################################################################
    # AB HIER MUSS DER EBEN ERZEUGTE DATAFRAME EINGEBUNDEN WERDEN! ##############################
    filepath = 'core/uploadStorage/EKPO_labeled_2021-09-25_17-31.xlsx'  # muss auskommentiert werden
    #pd.DataFrame()
    """Für CSV-Files"""
    # df = pd.read_csv(filepath, sep=";")
    """Für Excel-Files"""
    #df = pd.read_excel(filepath, engine='openpyxl')

    # start algorithm
    [accuracy, conf_matr, class_rep, y_pred, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, df_only_frauds] = mlalgo_func(filepath)

    # NEW CODE(Text Elements for ML_HTML PAGE)##############################################
    count_fraud = np.count_nonzero(y_pred==1)
    print("Number of Frauds: ")
    print(count_fraud)
    count_fraud_text = "From the selected data the following amount of fraud cases were detected: "

    count_nonfraud = np.count_nonzero(y_pred==0)
    print("Number of Non-Fraud in tested Data:")
    print(count_nonfraud)
    count_nonfraud_text = "The amount of non fraud cases are: "

    precision_text = "The self-evaluation of the Algorithm predicts an Accuracy of: "
    fraudtable_text = "The following Table shows the detected Fraud Cases"
    ####################################################################################
    # Shift to Frontend
    context = {
        "accuracy": accuracy,
        "conf_matr": conf_matr,
        "class_rep": class_rep,
        "y_pred": y_pred,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "data": df_only_frauds.to_html(classes="display table table-striped table-hover",
                                       table_id="dataShowTable_frauds", index=False,
                                     justify="center", header=True,),
        # Folgenden Eintrag ggf. noch einfügen ########################################
        #"datatable_names": datatable_names

        # NEW TEXT ELEMENTS FOR HTML PAGE ################################################
        "count_fraud": count_fraud,
        "count_fraud_text": count_fraud_text,
        "count_nonfraud": count_nonfraud,
        "count_nonfraud_text": count_nonfraud_text,
        "precision_text": precision_text,
        "fraudtable_text": fraudtable_text,
        }

    return render(httprequest, "myTemplates/machine-learning.html", context)

def mlalgo_func(filepath):
    """Call of the single functions for the Machine Learning part"""
    # start preprocessing (normalization, one-hot-encoding)
    [X_train, y_train, X_test, y_test] = prepro_func(filepath)
    print('Ausgabe der Y-Train Daten und Y-Test nach prepro:\n')
    print('Y-Train:',y_train)
    print('Y-Test:',y_test)

    # start alorithm KNN (K-nearest-neighbor)
    [y_pred, X_train, y_train, X_test, y_train_pred, y_test_pred] = mlalgo_knn(X_train,y_train,X_test)

    # df_only_frauds = pd.read_pickle('X_test.pkl')
    df_only_frauds = pd.read_pickle('dataframe_before_datatyp_check.pkl')  # reload created dataframe
    print('Read pickle-File...')
    print(df_only_frauds)
    # df_only_frauds['index'] = range(0, len(df))
    # df_only_frauds = df_only_frauds.set_index('index', inplace=True)

    print("Die Anzahl von Predict:", len(y_pred))
    index = 0

    for element in y_pred:

        if index >= len(y_pred):
            break
        # print('Inhalt des Elements: ',element)
        if element == 0.0:
            print("Kein Fraud")
            #df_only_frauds = df_only_frauds.drop([index], inplace=True)
            df_only_frauds.drop(index=[index], inplace=True)
            # df_only_frauds.drop((df_only_frauds.iloc[index, :]), inplace=True)
            # df_only_frauds = df_only_frauds.drop(index, axis=0)
            index = index + 1
        else:
            print("Fraud erkannt!")
            index = index + 1
    print(df_only_frauds)

    accuracy=metrics.accuracy_score(y_test, y_pred)
    conf_matr=confusion_matrix(y_test, y_pred)
    class_rep=classification_report(y_test, y_pred)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Test to see the content of Variable
    print(X_test)
    return accuracy, conf_matr, class_rep, y_pred, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, df_only_frauds


def prepro_func(input_file):
    """ Preprocesing of the data:
    ~ change X to 1 in Anomalie
    ~ remove empty columns
    ~"""


    #print('Python: {}'.format(sys.version))
    #print('Numpy:{}'.format(np.__version__))
    #print('Pandas:{}'.format(pd.__version__))
    #print('Matplotlib:{}'.format(plt.__version__))
    #print('Seaborn:{}'.format(sns.__version__))
    #print('Scipy:{}'.format(scipy.__version__))
    #print('Sklearn:{}'.format(sklearn.__version__))


    #ohe = OneHotEncoder(sparse=False)

    """Read File"""
    #df_fraud = pd.read_excel('output_labeled.xlsx')
    df_fraud = pd.read_excel(input_file, engine='openpyxl')

    print(df_fraud['Anomalie'])

    # change X to 1 and NaN to 0 of column 'Anomalie'
    df_fraud=prepro_anomalie_func(df_fraud)

    #remove NaN
    df_fraud = df_fraud.fillna(0)  # NaN oder Not a Number entfernt

    # remove unuseable columns for one-hot-encoding
    df_fraud_prepro = df_fraud.drop(
        ['Einkaufsbeleg', 'Position', 'Letzte Änderung am', 'Buchungskreis', 'Werk', 'Warengruppe', 'Einkaufsinfosatz',
         'Mengenumrechnung', 'Mengenumrechnung.1', 'entspricht', 'Nenner', 'Preiseinheit', 'InfoUpdate', 'Preisdruck',
         'Wareneingang', 'Rechnungseingang', 'Preisdatum', 'Einkaufsbelegtyp', 'FortschreibGruppe', 'Planlieferzeit',
         'Gewichtseinheit', 'Steuerstandort', 'Profitcenter', 'Übermittlungsuhrzeit', 'Nächste Übermittlg-Nr.',
         'Materialart', 'Zeitz. empf. St.ort', 'Periodenkennz. MHD', 'Bestellanforderung', 'Anforderer',
         'Endlieferung'], axis=1)

    # copy of the df without the ohe ##################################################################
    # df_fraud_prepro_copy = df_fraud_prepro[
    #    'Kurztext', 'Material', 'Material.1', 'Bestellmengeneinheit', 'BestellpreisME',
    #    'Basismengeneinheit']

    # one-hot-encode some columns
    #ohe.fit_transform(
     #   df_fraud_prepro[['Kurztext', 'Material', 'Material.1', 'Bestellmengeneinheit', 'BestellpreisME', 'Basismengeneinheit']])

    categorical_columns = ['Kurztext', 'Material', 'Material.1', 'Bestellmengeneinheit', 'BestellpreisME',
                           'Basismengeneinheit']
    encoder = ce.OneHotEncoder(cols=categorical_columns, use_cat_names=True)
    df_encoded = encoder.fit_transform(df_fraud_prepro)

    print('One-Hot-Encoder erfolgreich!')
    print(df_encoded)
    #print(df_encoded["Kurztext_Raisins"])

    #Split into test and training data, drop column Anomalie first
    y = df_encoded["Anomalie"]
    X = df_encoded.drop('Anomalie', axis=1)

    # Use random-state = 1, if you want each split to have equal results!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    # Do same steps without ohe for Fraud Datatable ################################################
    # y2 = df_fraud_prepro_copy["Anomalie"]
    # x2 = df_fraud_prepro_copy('Anomalie', axis=1)
    # x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=27)

    # Make pickle file for X_Test Data -> already ohe -> need different dataframe ###################
    # output_test = open('X_test.pkl', 'wb')
    # pickle.dump(x_test2, output_test)
    # output.close()

    #print(X_train)
    #print(X_test)
    #print(y_train)
    #print(y_test)
    #X_train.sample(5)




    scaler = StandardScaler().fit(X_train)
    #print(scaler)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    #scaler.mean_
    #scaler.scale_
    scaler.transform(X_train)
    X_train_scaled = scaler.transform(X_train)
    # print(X_train_scaled)
    #print(X_train_scaled.mean(axis=0))
    #print(X_train_scaled.std(axis=0))
    #X_test.head()

    scaler = StandardScaler().fit(X_test)
    scaler.mean_
    scaler.scale_
    scaler.transform(X_test)

    ####### Where is it used??
    X_test_scaled = scaler.transform(X_test)

    print(X_test_scaled.mean(axis=0))
    print(X_test_scaled.std(axis=0))

    return X_train, y_train, X_test, y_test





def mlalgo_knn(X_train, y_train, X_test):
    try: knn = pickle.load(open('knn_model', 'rb'))
    except: knn = KNeighborsClassifier(n_neighbors=7)
    # knn will be our ml model to train
    # knn = KNeighborsClassifier(n_neighbors=7)

    # train our model with training and target data
    knn.fit(X_train, y_train)

    # let the model predict a result with test data
    y_pred = knn.predict(X_test)
    print('Algorithmus erfolgreich angewendet!')

    # check accuracy of the model on the test data -> y_test needed for this
    # knn.score(X_test, y_test)

    # make pickle file:
    # write python dict to a file
    # mydict = {'a': 1, 'b': 2, 'c': 3}
    mydict = knn
    output = open('knn_model', 'wb')
    pickle.dump(mydict, output)
    output.close()

    # read python dict back from the file
    # pkl_file = open('knn_model', 'rb')
    # mydict2 = pickle.load(pkl_file)
    # pkl_file.close()

    # print(mydict)
    # print(mydict2)

    # https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py 13.08.21
    # get the prediction labels and outlier scores of the training data
    y_train_pred = knn.predict(X_test) # knn.labels_  # binary labels (0: inliers, 1: outliers)
   # y_train_scores = knn.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = knn.predict(X_test)  # outlier labels (0 or 1)
    # y_test_scores = knn.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    # print("\nOn Training Data:")
    # evaluate_print('KNN', y_train, y_train_scores)
    # print("\nOn Test Data:")
    # evaluate_print('KNN', y_test, y_test_scores)

    # visualize the results
    # visualize('KNN', X_train, y_train, X_test, y_test, y_train_pred,
             # y_test_pred, show_figure=True, save_figure=True)

    return y_pred, X_train, y_train, X_test, y_train_pred, y_test_pred


def prepro_anomalie_func(transfered_data_frame):
    """Change X's of column 'Anomalie' to 1
    and empty cells to 0 for normalization"""

    df_fraud=transfered_data_frame
    index = 0
    for val in df_fraud['Anomalie']:
        #print('For-Schleife Wer von Val:', val, index)
        if val == 'x':
            #print('Value is X')
            #df_fraud['Anomalie'][index] = val
            df_fraud.loc[index, 'Anomalie']= 1.0
            #print(df_fraud.loc[index, 'Anomalie'])
            index += 1
            #print(index)

        # value is NaN
        else:
            val = 0.0
            #df_fraud['Anomalie'][index] = val
            df_fraud.loc[index, 'Anomalie'] = 0.0
            #print(df_fraud.loc[index, 'Anomalie'])
            index += 1
            #print(index)



    print('Anomalie erfolgreich encoded!')
    #print(df_fraud['Anomalie'])
    return df_fraud

# Analyze-file func will not train the model, just analyze


def analyze_file(httprequest, *args, **kwargs):
    try: knn = pickle.load(open('knn_model', 'rb'))
    except: knn = KNeighborsClassifier(n_neighbors=7)

    datafile = 'core/uploadStorage/EKPO_labeled_2021-09-25_17-31.xlsx'
    df_fraud = pd.read_excel(datafile, engine='openpyxl')

    # remove NaN
    df_fraud = df_fraud.fillna(0)  # NaN oder Not a Number entfernt

    # remove unuseable columns for one-hot-encoding
    df_fraud_prepro = df_fraud.drop(
        ['Einkaufsbeleg', 'Position', 'Letzte Änderung am', 'Buchungskreis', 'Werk', 'Warengruppe', 'Einkaufsinfosatz',
         'Mengenumrechnung', 'Mengenumrechnung.1', 'entspricht', 'Nenner', 'Preiseinheit', 'InfoUpdate', 'Preisdruck',
         'Wareneingang', 'Rechnungseingang', 'Preisdatum', 'Einkaufsbelegtyp', 'FortschreibGruppe', 'Planlieferzeit',
         'Gewichtseinheit', 'Steuerstandort', 'Profitcenter', 'Übermittlungsuhrzeit', 'Nächste Übermittlg-Nr.',
         'Materialart', 'Zeitz. empf. St.ort', 'Periodenkennz. MHD', 'Bestellanforderung', 'Anforderer',
         'Endlieferung'], axis=1)
    categorical_columns = ['Kurztext', 'Material', 'Material.1', 'Bestellmengeneinheit', 'BestellpreisME',
                           'Basismengeneinheit']
    encoder = ce.OneHotEncoder(cols=categorical_columns, use_cat_names=True)
    df_encoded = encoder.fit_transform(df_fraud_prepro)

    # Scale the encoded dataframe
    scaler = StandardScaler().fit(df_encoded)
    df_analyze = scaler.transform(df_encoded)

    # Predict the dataframe with the ml model
    y_pred = knn.predict(df_analyze)
    knn.close()

    # Shift results to Frontend
    context = {
        # "accuracy": accuracy,
        # "conf_matr": conf_matr,
        # "class_rep": class_rep,
        "y_pred": y_pred,
        # "data": df_only_frauds.to_html(classes="display table table-striped table-hover",
        #                               table_id="dataShowTable_frauds", index=False,
        #                               justify="center", header=True, )
    }

    return render(httprequest, "myTemplates/machine-learning.html", context)
