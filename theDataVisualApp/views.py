from django.shortcuts import render

# Create your views here.

def datavisu_view(httprequest, *args, **kwargs):
    """Do  anything with request"""
    return render(httprequest, "myTemplates/data-visualization.html")


'''
#Vorbereitung Datenvisualisierung (https://www.chartjs.org/docs/latest/getting-started/integration.html)
#npm
npm install chart.js

#Script Tag
< script
src = "path/to/chartjs/dist/chart.js" > < / script >
< script >
var
myChart = new
Chart(ctx, {...});
< / script >


#Common JS
var Chart = require('chart.js');
var myChart = new Chart(ctx, {...});


# Bundlers (Webpack, Rollup, etc.)
import Chart from 'chart.js/auto';

# Require JS
require(['path/to/chartjs/dist/chart.min.js'], function(Chart){
    var myChart = new Chart(ctx, {...});
});

require(['chartjs'], function(Chart) {
    require(['moment'], function() {
        require(['chartjs-adapter-moment'], function() {
            new Chart(ctx, {...});
        });
    });
});

#Vorbereitung Installation Datenvisualisierung (Video)

var myChartObject = document.getElementById('myChart');

var chart = new Chart(myChartObject, {
    type: 'pie',
    data: {
        labels: ["Fraud", "not-Fraud"],
        datasets: [{
            label: "input_file"
            data: [1,]
'''