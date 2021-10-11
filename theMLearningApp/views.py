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
import json
from django.http import HttpResponse


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
    """Show startpage for starting Machine Learning algorithm for specific data set"""
    return render(httprequest, "myTemplates/machine-learning.html")



def train_mlalgo(httprequest, *args, **kwargs):
    """ Main function for Machine Learning Algorithm
            Call of the single functions for the Machine Learning part
            dataframe from datatbase is transfered to function
        """
    # takes the selection from frontend and saves it to selected_option
    if httprequest.POST:  # If this is true, the view received POST
        selected_table = httprequest.POST.get('select_df', None)
        dataframe_from_sql=read_table_from_sql(selected_table)

        """
        #############################################################################################
        # AB HIER MUSS DER EBEN ERZEUGTE DATAFRAME EINGEBUNDEN WERDEN! ##############################
        filepath = 'core/uploadStorage/EKPO_labeled_2021-09-25_17-31.xlsx'  # muss auskommentiert werden
        #pd.DataFrame()
        #ür CSV-Files
        # df = pd.read_csv(filepath, sep=";")
        #Für Excel-Files
        #df = pd.read_excel(filepath, engine='openpyxl')
        ##############################################################################################
        """

        # start algorithm
        #[accuracy, conf_matr, class_rep, y_pred, df_only_frauds] = mlalgo_func(dataframe_from_sql)

        # start preprocessing (normalization, one-hot-encoding)
        [X_train, y_train, X_test, y_test, X_train_index, X_test_index] = prepro_func(dataframe_from_sql, analyze=False)
        print('Ausgabe der Y-Train Daten und Y-Test nach prepro:\n')
        print('Y-Train:', y_train)
        count_fraud = np.count_nonzero(y_train == 1)
        print("Number of Frauds in y_train: ")
        print(count_fraud)
        print('Y-Test:', y_test)
        count_fraud = np.count_nonzero(y_test == 1)
        print("Number of Frauds in y_test: ")
        print(count_fraud)

        # start alorithm KNN (K-nearest-neighbor)
        y_pred = mlalgo_knn(X_train, y_train, X_test)

        # call show_predicted_frauds function to get dataframe with predicted frauds from dataset
        df_only_frauds = show_predicted_frauds(y_pred, X_test_index)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        conf_matr = confusion_matrix(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

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
            #"X_train": X_train,
            #"y_train": y_train,
            #"X_test": X_test,
            #"y_test": y_test,
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

    else:
        context = {
            "error" : true,
            "message": "No POST-Request identified",
        }

    return render(httprequest, "myTemplates/machine-learning.html", context)

def analyze_file(httprequest, *args, **kwargs):
    """ Analyze-file func will not train the model, just analyze"""
    try: knn = pickle.load(open('knn_model', 'rb'))
    except: context = {"error": true,}

    if httprequest.POST:  # If this is true, the view received POST
        print("Start analysing file...")
        selected_table = httprequest.POST.get('select_df', None)
        dataframe_from_sql=read_table_from_sql(selected_table)

        # start preprocessing (normalization, one-hot-encoding)
        [X_test, X_test_index] = prepro_func(dataframe_from_sql, analyze=True)

        # Predict the dataframe with the ml model
        y_pred = knn.predict(X_test)
        print('Algorithmus erfolgreich angewendet!')

        # call show_predicted_frauds function to get dataframe with predicted frauds from dataset
        df_only_frauds = show_predicted_frauds(y_pred, X_test_index)

        # NEW CODE(Text Elements for ML_HTML PAGE)##############################################
        count_fraud = np.count_nonzero(y_pred == 1)
        print("Number of Frauds: ")
        print(count_fraud)
        count_fraud_text = "From the selected data the following amount of fraud cases were detected: "

        count_nonfraud = np.count_nonzero(y_pred == 0)
        print("Number of Non-Fraud in tested Data:")
        print(count_nonfraud)
        count_nonfraud_text = "The amount of non fraud cases are: "

        precision_text = "The self-evaluation of the Algorithm predicts an Accuracy of: "
        fraudtable_text = "The following Table shows the detected Fraud Cases"
        ####################################################################################

        # Shift results to Frontend
        context = {

            "y_pred": y_pred,
             "data": df_only_frauds.to_html(classes="display table table-striped table-hover",
                                           table_id="dataShowTable_frauds", index=False,
                                           justify="center", header=True, ),
            "count_fraud": count_fraud,
            "count_fraud_text": count_fraud_text,
            "count_nonfraud": count_nonfraud,
            "count_nonfraud_text": count_nonfraud_text,
            "fraudtable_text": fraudtable_text,
        }
    else:
        context = {
            "error": true,
            "message": "No POST-Request identified",
        }

    return render(httprequest, "myTemplates/machine-learning.html", context)

def prepro_func(dataframe_from_sql, analyze):
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
    #df_fraud = pd.read_excel(input_file, engine='openpyxl')
    """Receive dataframe from SQL: zeros and empty columns should be removed and datatypes are checked and modified by user via show_data_view
        dataframe is now ready for machine learning"""

    if not analyze:  #only for training
        print(dataframe_from_sql['Anomalie'])
        # change X to 1 and NaN to 0 of column 'Anomalie'
        dataframe_from_sql=prepro_anomalie_func(dataframe_from_sql)
    else:
        if 'Anomalie' in dataframe_from_sql.columns:
            dataframe_from_sql=dataframe_from_sql.drop('Anomalie', axis=1)


    #remove NaN
    dataframe_from_sql = dataframe_from_sql.fillna(0)  # NaN oder Not a Number entfernt

    # remove unuseable columns for one-hot-encoding
    df_fraud_prepro = dataframe_from_sql.drop(
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

    print('One-Hot-Encoder erfolgreich! Das ist der OHE-Dataframe:')
    print(df_encoded)
    #print(df_encoded["Kurztext_Raisins"])

    # Add index again
    #df_encoded= pd.concat([df_fraud_index, df_encoded], axis=1)

    if not analyze:  #only for training
        #Split into test and training data, drop column Anomalie first
        y = df_encoded["Anomalie"]
        X = df_encoded.drop('Anomalie', axis=1)
        # Use random-state = 1, if you want each split to have equal results!
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27, stratify=y)

        """
            Now we start to scale the data to values between -1 and 1 to make them compareable
        """

        """ OLD TRY TRAIN
            #scaler = StandardScaler().fit(X_train)
            #print(scaler)
            StandardScaler(copy=True, with_mean=True, with_std=True)
            #scaler.mean_
            #scaler.scale_
            #scaler.transform(X_train)
            X_train_scaled = scaler.transform(X_train)
            print('Ausgabe X_train_scaled')
            print(X_train_scaled)
            #print(X_train_scaled.mean(axis=0))
            #print(X_train_scaled.std(axis=0))
            #X_test.head()
        """

        #Remove index and add later after KNN Algorithm - not necessary
        X_train_index=X_train["index"]
        X_train_noindex = X_train.drop('index', axis=1)

        # Remove index and add later after KNN Algorithm to identify frauds via y_pred
        X_test_index = X_test["index"]
        X_test_noindex = X_test.drop('index', axis=1)

        scaler_train = StandardScaler()  # initialize scaler
        X_train_scaled = scaler_train.fit_transform(
            X_train_noindex)  # fit the data to values between -1 and 1 and than transform them
        print("Die Durchschnittswerte: ", scaler_train.mean_, "Skalierung: ",
              scaler_train.scale_)  # calculate the mean value of the single column
        print("X_train_scaled:", X_train_scaled)

        """ OLD TRY TEST
            scaler = StandardScaler().fit(X_test)
            scaler.mean_
            scaler.scale_
            scaler.transform(X_test)

            ####### Where is it used??
            X_test_scaled = scaler.transform(X_test)
        """

        scaler_test = StandardScaler()  # initialize scaler
        X_test_scaled = scaler_test.fit_transform(
            X_test_noindex)  # fit the data to values between -1 and 1 and than transform them
        print("Die Durchschnittswerte: ", scaler_test.mean_, "Skalierung: ",
              scaler_test.scale_)  # calculate the mean value of the single column
        print("X_test_scaled:", X_test_scaled)

        """
        print('Ausgabe der skalieren Daten (mean):')
        print(X_test_scaled.mean(axis=0))
        print('(std):')
        print(X_test_scaled.std(axis=0))
        """

        return X_train_scaled, y_train, X_test_scaled, y_test, X_train_index, X_test_index

    else:
        X_test=df_encoded  #gesamter Datensatz verwenden zum Testen
        # Remove index and add later after KNN Algorithm to identify frauds via y_pred
        X_test_index = X_test["index"]
        X_test_noindex = X_test.drop('index', axis=1)

        scaler_test = StandardScaler()  # initialize scaler
        X_test_scaled = scaler_test.fit_transform(X_test_noindex)  # fit the data to values between -1 and 1 and than transform them
        print("Die Durchschnittswerte: ", scaler_test.mean_, "Skalierung: ",
              scaler_test.scale_)  # calculate the mean value of the single column
        print("X_test_scaled:", X_test_scaled)


        """
        print('Ausgabe der skalieren Daten (mean):')
        print(X_test_scaled.mean(axis=0))
        print('(std):')
        print(X_test_scaled.std(axis=0))
        """

        return  X_test_scaled, X_test_index

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
    print(df_fraud['Anomalie'])
    return df_fraud

def mlalgo_knn(X_train, y_train, X_test):
    #try: knn = pickle.load(open('knn_model', 'rb'))
    #except: knn = KNeighborsClassifier(n_neighbors=7)
    # knn will be our ml model to train
    knn = KNeighborsClassifier(n_neighbors=5)

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
    mydict = knn   #safe model as pickle file for analyzing
    output = open('knn_model', 'wb')
    pickle.dump(mydict, output)
    output.close()

    """
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
             
             """

    return y_pred


def find_datatables(request):
    """AJAX Request for select-box, to get all datatables"""

    if request.is_ajax():
        # GET ALL TABLES FROM DATABASE (for frontend dropdown selection)
        conn = sqlite3.connect('TestDB1.db')
        c = conn.cursor()
        c.execute('''SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC ''')
        datatable_names = c.fetchall()
        conn.close()

    else:
        datatable_names = "NO DATA"

    data=json.dumps(datatable_names)
    return HttpResponse(data, content_type='applicationj/json')

def read_table_from_sql(selected_table):
    print("################################################################################")
    print("Der ausgewählte DF ist: " + selected_table)

    # Read Dataframe from Database
    conn = sqlite3.connect('TestDB1.db')
    dataframe_from_sql = pd.read_sql('SELECT * FROM {}'.format("" + selected_table + ""), conn)
    print("The selected Dataframe from Frontend is:")
    print(dataframe_from_sql)

    # create *.pkl from the selected datatable for use in Visualisation
    dataframe_from_sql.to_pickle('dataframe_from_sql_database.pkl')
    return dataframe_from_sql

def show_predicted_frauds(y_pred, X_test_index):
    """ create Dataframe with y_pred and the connected index to find frauds in database"""
    # add index column after KNN is finished  - FIRST TRY, DELETE LATER
    # df_encoded= pd.concat([df_fraud_index, df_encoded], axis=1)
    # X_test = pd.concat([X_test_index, X_test], axis=1)
    # y_pred = pd.concat([X_test_index, y_pred], axis=1)
    y_pred_with_index = pd.DataFrame({'index': X_test_index, 'y_pred': y_pred})

    print(y_pred_with_index)

    df_from_sql_database = pd.read_pickle('dataframe_from_sql_database.pkl')

    """Zeile einlesen,
     schauen ob bei y_pred 1 steht
      -> index der Zeile bestimmen ->
       Zeile mit diesem Index aus dataframe auslesen
       Zeile ausgeben"""

    df_only_frauds = pd.DataFrame(columns=df_from_sql_database.columns)

    for index, row in y_pred_with_index.iterrows():
        # print(int(row['index']), 'Y_PRED: ', row['y_pred'])
        if row['y_pred'] == 1:
            print('\nIndex der Zeile: ', int(row['index']), '\n')
            # df_from_sql_database
            # print(X_test.index(element.index))
            line_with_fraud = df_from_sql_database.loc[[int(row['index'])]]
            print(line_with_fraud)
            df_only_frauds = df_only_frauds.append(line_with_fraud, ignore_index=True)

    print(df_only_frauds)



    """
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
    """
    return df_only_frauds


