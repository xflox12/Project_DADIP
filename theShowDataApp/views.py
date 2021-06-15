from django.shortcuts import render
import sqlite3

# Create your views here.

def showdata_view(httprequest, *args, **kwargs):
    """Do  anything with request"""

    conn = sqlite3.connect('TestDB1.db')
    c = conn.cursor()

    c.execute('''  
        SELECT * FROM FRAUDS
                  ''')

    showData=c.fetchall()
    context = {
        'showData': showData
    }

    return render(httprequest, "myTemplates/showdata.html", context)